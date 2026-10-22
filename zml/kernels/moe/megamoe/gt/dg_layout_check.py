"""Cross-check gt/layout_gt.json (DeepGEMM headers compiled on host) against DeepGEMM's compiled
library (vLLM's _C extension) on the current GPU: get_symm_buffer_size_for_mega_moe and
get_block_m_for_mega_moe."""
import json, sys
shim, gt_path = sys.argv[1], sys.argv[2]
sys.path.insert(0, shim)
import torch
from deep_gemm import _C
gt = json.load(open(gt_path))
sms = torch.cuda.get_device_properties(0).multi_processor_count
print('num_sms', sms)
n = bad = 0
for b in gt['buffers']:
    if b['num_sms'] != sms:
        continue
    nbytes, info = _C.get_symm_buffer_size_for_mega_moe(b['num_ranks'], b['num_experts'], b['num_max_tokens_per_rank'], b['num_topk'], b['hidden'], b['intermediate'], 'fp8xfp4', 'swiglu', b['num_shared_experts'])
    n += 1
    if nbytes != b['layout'][0]:
        bad += 1; print('MISMATCH bytes', b, nbytes)
    # the layout_info list is captured inside the closure; check via slicing a fake buffer
print('buffers checked', n, 'mismatches', bad)
n = bad = 0
for c in gt['configs']:
    R, E, T, K, ntok = c[0], c[1], c[2], c[3], c[4]
    bm = _C.get_block_m_for_mega_moe(R, E, T, ntok, K, 'fp8xfp4')
    n += 1
    if bm != c[6]:
        bad += 1; print('MISMATCH block_m', c, bm)
print('block_m checked', n, 'mismatches', bad)
print('token alignment', _C.get_token_alignment_for_mega_moe())
