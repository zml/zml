"""Run DeepGEMM-function reference micro-kernels (numerics_ref.cu) on random inputs and save
inputs + outputs as raw little-endian files for zml/kernels/moe/megamoe/gpu_test.zig.
Usage: python -I numerics_ref.py <deep_gemm include dir> <out dir>"""
import os, sys, json
import numpy as np
import torch
from torch.utils.cpp_extension import load_inline

inc, out = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(out, 'nvcc_keep'), exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))
cuda_src = open(os.path.join(here, 'numerics_ref.cu')).read()
cuda_src += r'''
#include <torch/extension.h>
void l1(torch::Tensor raw, torch::Tensor w, torch::Tensor o, torch::Tensor a, bool fm, bool cl) {
    launch_l1((const uint32_t*) raw.data_ptr(), (const float*) w.data_ptr(), (uint32_t*) o.data_ptr(), (uint32_t*) a.data_ptr(), (int) w.numel() / 2, fm, cl);
}
void l2(torch::Tensor raw, torch::Tensor o) { launch_l2((const uint32_t*) raw.data_ptr(), (uint32_t*) o.data_ptr(), (int) o.numel() / 4); }
void comb(torch::Tensor s, torch::Tensor n, torch::Tensor o) { launch_combine((const uint32_t*) s.data_ptr(), (const int*) n.data_ptr(), (uint32_t*) o.data_ptr(), (int) o.numel()); }
void maps(torch::Tensor a, torch::Tensor ao, torch::Tensor b, torch::Tensor bo, torch::Tensor to) { launch_maps((const uint32_t*) a.data_ptr(), (uint32_t*) ao.data_ptr(), (const uint32_t*) b.data_ptr(), (uint32_t*) bo.data_ptr(), (uint32_t*) to.data_ptr()); }
'''
cpp_src = '''
void l1(torch::Tensor raw, torch::Tensor w, torch::Tensor o, torch::Tensor a, bool fm, bool cl);
void l2(torch::Tensor raw, torch::Tensor o);
void comb(torch::Tensor s, torch::Tensor n, torch::Tensor o);
void maps(torch::Tensor a, torch::Tensor ao, torch::Tensor b, torch::Tensor bo, torch::Tensor to);
'''
mod = load_inline('megamoe_numerics_ref', cpp_sources=cpp_src, cuda_sources=cuda_src, functions=['l1', 'l2', 'comb', 'maps'],
                  extra_include_paths=[inc], extra_cuda_cflags=['-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_BFLOAT16_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__', '-O3', '-std=c++20', '--expt-relaxed-constexpr',
                  '-gencode=arch=compute_103a,code=sm_103a', '--keep', '--keep-dir', os.path.join(out, 'nvcc_keep')],
                  build_directory=None, verbose=False)
rng = np.random.default_rng(1234)
N = 128 * 64

def mixed(shape):
    scales = np.array([1e-4, 0.01, 0.1, 1, 3, 9.9, 10, 10.1, 30, 1e3, 1e6], np.float32)
    v = rng.standard_normal(shape).astype(np.float32) * scales[rng.integers(0, len(scales), shape)]
    specials = np.array([0.0, -0.0, 1e-40, -1e-40, 3e38, -3e38, np.inf, -np.inf, np.nan, 10.0, -10.0, 10.0390625], np.float32)
    mask = rng.random(shape) < 0.02
    v[mask] = specials[rng.integers(0, len(specials), int(mask.sum()))]
    return v

raw = mixed((N, 8))
w = rng.random((N, 2)).astype(np.float32)
m = rng.random((N, 2)) < 0.02
w[m] = np.array([0.0, 1e-30, 1.0, 0.5], np.float32)[rng.integers(0, 4, int(m.sum()))]
raw.view(np.uint32).tofile(os.path.join(out, 'l1_raw.bin'))
w.tofile(os.path.join(out, 'l1_w.bin'))
for fm in (1,):
    for cl in (1, 0):
        o = torch.zeros(N * 4, dtype=torch.int32, device='cuda')
        a = torch.zeros(N * 4, dtype=torch.int32, device='cuda')
        mod.l1(torch.from_numpy(raw.view(np.int32)).cuda(), torch.from_numpy(w).cuda(), o, a, bool(fm), bool(cl))
        torch.cuda.synchronize()
        o.cpu().numpy().tofile(os.path.join(out, f'l1_out_fm{fm}_cl{cl}.bin'))
        a.cpu().numpy().tofile(os.path.join(out, f'l1_act_fm{fm}_cl{cl}.bin'))

l2raw = mixed((N, 8))
l2raw.view(np.uint32).tofile(os.path.join(out, 'l2_raw.bin'))
o = torch.zeros(N * 4, dtype=torch.int32, device='cuda')
mod.l2(torch.from_numpy(l2raw.view(np.int32)).cuda(), o)
o.cpu().numpy().tofile(os.path.join(out, 'l2_out.bin'))

# combine slots: bf16x2 words from bf16-rounded mixed values
vals = torch.from_numpy(mixed((N, 7, 2))).to(torch.bfloat16).view(torch.int16).numpy().astype(np.uint16)
slots = (vals[..., 0].astype(np.uint32) | (vals[..., 1].astype(np.uint32) << 16))
nslots = rng.integers(0, 8, N).astype(np.int32)
slots.tofile(os.path.join(out, 'comb_slots.bin'))
nslots.tofile(os.path.join(out, 'comb_n.bin'))
o = torch.zeros(N, dtype=torch.int32, device='cuda')
mod.comb(torch.from_numpy(slots.view(np.int32)).cuda(), torch.from_numpy(nslots).cuda(), o)
o.cpu().numpy().tofile(os.path.join(out, 'comb_out.bin'))
l1v = rng.integers(0, 2**32, 128 * 4, dtype=np.uint64).astype(np.uint32)
l2v = rng.integers(0, 2**32, 128 * 4 * 4, dtype=np.uint64).astype(np.uint32)
ao = torch.zeros(32 * 64 // 4, dtype=torch.int32, device='cuda')
bo = torch.zeros(2 * 32 * 128 // 4, dtype=torch.int32, device='cuda')
to = torch.zeros(128 * 2 * 8, dtype=torch.int32, device='cuda')
mod.maps(torch.from_numpy(l1v.view(np.int32)).cuda(), ao, torch.from_numpy(l2v.view(np.int32)).cuda(), bo, to)
torch.cuda.synchronize()
l1v.tofile(os.path.join(out, 'stsm_l1_vals.bin')); ao.cpu().numpy().tofile(os.path.join(out, 'stsm_l1_out.bin'))
l2v.tofile(os.path.join(out, 'stsm_l2_vals.bin')); bo.cpu().numpy().tofile(os.path.join(out, 'stsm_l2_out.bin'))
to.cpu().numpy().tofile(os.path.join(out, 'tmem_ld_out.bin'))
json.dump({'n': N, 'device': torch.cuda.get_device_name(), 'sms': torch.cuda.get_device_properties(0).multi_processor_count}, open(os.path.join(out, 'meta.json'), 'w'))
print('ok', N)
