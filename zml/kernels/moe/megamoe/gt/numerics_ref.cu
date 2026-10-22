// Reference micro-kernels for the MegaMoE port's numerics, built from DeepGEMM's own device
// functions (vLLM's vendored headers). The L1 epilogue block is copied verbatim from
// impls/sm100_fp8_fp4_mega_moe.cuh (SwiGLU, amax, UE8M0, FP8 cast); only the surrounding
// loads/stores differ. Compiled like DeepGEMM's JIT: nvcc -O3, no --use_fast_math.
// torch.utils.cpp_extension defines these; DeepGEMM's JIT does not.
#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__
#define DG_IN_CUDA_COMPILATION
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cute/numeric/numeric_types.hpp>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/ptx/utils.cuh>

using namespace deep_gemm;

// One thread = one TMEM 16x256b.x1 pair of loads (8 u32: gate/up for 2 channel pairs x 2 tokens).
// `kActivationClamp` < 0 encodes infinity.
template <bool kFastMath, bool kClamp>
__global__ void l1_epilogue_ref(const uint32_t* raw, const float* w, uint32_t* out, uint32_t* act_out) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lane_idx = ptx::get_lane_idx();
    constexpr float kActivationClamp = kClamp ? 10.0f : cute::numeric_limits<float>::infinity();
    constexpr bool kClampAfterSilu = false;
    constexpr float kActivationAlpha = 1.0f;
    constexpr float kActivationBeta = 0.0f;
    constexpr uint32_t kNumAtomsPerStore = 1;
    constexpr uint32_t i = 0;
    uint32_t store_block_raw_values[kNumAtomsPerStore][8];
    #pragma unroll
    for (uint32_t j = 0; j < 8; ++ j) store_block_raw_values[0][j] = raw[t * 8 + j];
    const float2 weights = {w[t * 2 + 0], w[t * 2 + 1]};
    float2 activation_values[kNumAtomsPerStore][2];
    float2 amax_values[kNumAtomsPerStore];

    // ---- verbatim from sm100_fp8_fp4_mega_moe.cuh (vLLM vendored), L1 epilogue ----
                        // Apply SwiGLU: gate * sigmoid(alpha * gate) * (up + beta)
                        auto fp32_values = reinterpret_cast<float2*>(store_block_raw_values[i]);
                        #pragma unroll
                        for (uint32_t k = 0; k < 2; ++ k) {
                            auto bf16_gate = __float22bfloat162_rn(fp32_values[k * 2 + 0]);
                            auto bf16_up =   __float22bfloat162_rn(fp32_values[k * 2 + 1]);

                            // Clamp
                            if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity()) {
                                if constexpr (not kClampAfterSilu)
                                    bf16_gate = __hmin2(bf16_gate, {kActivationClamp, kActivationClamp});
                                bf16_up = __hmax2(bf16_up, {-kActivationClamp, -kActivationClamp});
                                bf16_up = __hmin2(bf16_up, {kActivationClamp, kActivationClamp});
                            }

                            // SwiGLU
                            const auto gate = __bfloat1622float2(bf16_gate);
                            auto sigmoid_input = gate;
                            if constexpr (kActivationAlpha != 1.0f)
                                sigmoid_input = __fmul2_rn(
                                    sigmoid_input, {kActivationAlpha, kActivationAlpha});
                            auto neg_gate_exp = make_float2(
                                kFastMath ? __expf(-sigmoid_input.x) : expf(-sigmoid_input.x),
                                kFastMath ? __expf(-sigmoid_input.y) : expf(-sigmoid_input.y));
                            const auto denom = __fadd2_rn({1.0f, 1.0f}, neg_gate_exp);
                            float2 gated;
                            if constexpr (kFastMath) {
                                gated = __fmul2_rn(
                                    gate, {math::fast_rcp(denom.x), math::fast_rcp(denom.y)});
                            } else {
                                gated = {gate.x / denom.x, gate.y / denom.y};
                            }
                            if constexpr (kClampAfterSilu and kActivationClamp != cute::numeric_limits<float>::infinity())
                                gated = {fminf(gated.x, kActivationClamp), fminf(gated.y, kActivationClamp)};
                            auto up = __bfloat1622float2(bf16_up);
                            if constexpr (kActivationBeta != 0.0f)
                                up = __fadd2_rn(up, {kActivationBeta, kActivationBeta});
                            activation_values[i][k] = __fmul2_rn(
                                __fmul2_rn(gated, up), weights);
                        }

                        // Amax reduction (thread-level)
                        float2 thread_local_amax = {0.f, 0.f};
                        #pragma unroll
                        for (uint32_t k = 0; k < 2; ++ k) {
                            thread_local_amax.x = fmaxf(thread_local_amax.x, fabsf(activation_values[i][k].x));
                            thread_local_amax.y = fmaxf(thread_local_amax.y, fabsf(activation_values[i][k].y));
                        }

                        // Amax reduction (warp-level)
                        amax_values[i].x = math::warp_reduce<4, true>(
                            thread_local_amax.x, math::ReduceMax<float>());
                        amax_values[i].y = math::warp_reduce<4, true>(
                            thread_local_amax.y, math::ReduceMax<float>());
    // ---- (the cross-warp amax via shared memory is a plain fmaxf of two warps' values) ----
                        // Calculate SF
                        const uint2 sf_exp = {math::get_ue8m0_sf_exp(amax_values[i].x),
                                              math::get_ue8m0_sf_exp(amax_values[i].y)};
                        const float2 sf_inv = {math::get_ue8m0_sf_inv<float>(sf_exp.x),
                                               math::get_ue8m0_sf_inv<float>(sf_exp.y)};

                        // Cast
                        const float2 upper = __fmul2_rn(activation_values[i][0], sf_inv);
                        const float2 lower = __fmul2_rn(activation_values[i][1], sf_inv);
                        const auto fp8x4_values = __nv_fp8x4_e4m3(make_float4(upper.x, upper.y, lower.x, lower.y));
    // ---- end verbatim ----
    (void) lane_idx;
    out[t * 4 + 0] = fp8x4_values.__x;
    out[t * 4 + 1] = sf_exp.x;
    out[t * 4 + 2] = sf_exp.y;
    out[t * 4 + 3] = __float_as_uint(amax_values[i].x);
    act_out[t * 4 + 0] = __float_as_uint(activation_values[i][0].x);
    act_out[t * 4 + 1] = __float_as_uint(activation_values[i][0].y);
    act_out[t * 4 + 2] = __float_as_uint(activation_values[i][1].x);
    act_out[t * 4 + 3] = __float_as_uint(activation_values[i][1].y);
}

// L2 epilogue: `math::cast_into_bf16_and_pack(values[2q], values[2q + 1])` for q = 0..3.
__global__ void l2_pack_ref(const uint32_t* raw, uint32_t* out) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t values[8];
    #pragma unroll
    for (uint32_t j = 0; j < 8; ++ j) values[j] = raw[t * 8 + j];
    #pragma unroll
    for (uint32_t q = 0; q < 4; ++ q)
        out[t * 4 + q] = math::cast_into_bf16_and_pack(values[2 * q], values[2 * q + 1]);
}

// Combine: `reduced` starts at 0, `ptx::accumulate` over the present slots in order, then
// `__float22bfloat162_rn` (impls/sm100_fp8_fp4_mega_moe.cuh combine loop).
__global__ void combine_ref(const uint32_t* slots, const int* num_slots, uint32_t* out) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    float2 reduced = {};
    for (int s = 0; s < num_slots[t]; ++ s) {
        const uint32_t word = slots[t * 7 + s];
        ptx::accumulate(reduced, *reinterpret_cast<const nv_bfloat162*>(&word));
    }
    const auto casted = __float22bfloat162_rn(reduced);
    out[t] = *reinterpret_cast<const uint32_t*>(&casted);
}

void launch_l1(const uint32_t* raw, const float* w, uint32_t* out, uint32_t* act, int n, bool fast_math, bool clamp) {
    const dim3 grid(n / 128), block(128);
    if (fast_math and clamp) l1_epilogue_ref<true, true><<<grid, block>>>(raw, w, out, act);
    if (fast_math and not clamp) l1_epilogue_ref<true, false><<<grid, block>>>(raw, w, out, act);
    if (not fast_math and clamp) l1_epilogue_ref<false, true><<<grid, block>>>(raw, w, out, act);
    if (not fast_math and not clamp) l1_epilogue_ref<false, false><<<grid, block>>>(raw, w, out, act);
}
void launch_l2(const uint32_t* raw, uint32_t* out, int n) { l2_pack_ref<<<n / 128, 128>>>(raw, out); }
void launch_combine(const uint32_t* slots, const int* num_slots, uint32_t* out, int n) { combine_ref<<<n / 128, 128>>>(slots, num_slots, out); }

// ---- register <-> shared/tensor memory maps (one epilogue warpgroup, 4 atoms) ----
#include <deep_gemm/ptx/ld_st.cuh>
#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/arch/barrier.h>

// L1: `SM100_U8x4_STSM_T` with the impl's 64B-swizzled address (row = lane, col = warp in WG).
__global__ void stsm_l1_ref(const uint32_t* vals, uint32_t* out) {
    __shared__ __align__(1024) uint8_t smem[32 * 64];
    const uint32_t lane_idx = threadIdx.x % 32, warp_idx_in_wg = threadIdx.x / 32;
    for (uint32_t j = threadIdx.x; j < 32 * 64 / 4; j += 128) reinterpret_cast<uint32_t*>(smem)[j] = 0xdeadbeef;
    __syncthreads();
    constexpr uint32_t ATOM_M = 8, L1_OUT_BLOCK_N = 64, kNumBankGroupBytes = 16;
    #pragma unroll
    for (uint32_t i = 0; i < 4; ++ i) {
        uint32_t row = lane_idx;
        uint32_t col = warp_idx_in_wg;
        const auto smem_ptr = smem + i * ATOM_M * L1_OUT_BLOCK_N + row * L1_OUT_BLOCK_N + (col ^ (row / 2)) * kNumBankGroupBytes;
        ptx::SM100_U8x4_STSM_T<uint32_t>::copy(vals[threadIdx.x * 4 + i], smem_ptr);
    }
    __syncthreads();
    for (uint32_t j = threadIdx.x; j < 32 * 64 / 4; j += 128) out[j] = reinterpret_cast<uint32_t*>(smem)[j];
}

// L2: `SM90_U32x4_STSM_T` with the impl's 128B-swizzled address.
__global__ void stsm_l2_ref(const uint32_t* vals, uint32_t* out) {
    __shared__ __align__(1024) uint8_t smem[2 * 32 * 128];
    const uint32_t lane_idx = threadIdx.x % 32, warp_idx_in_wg = threadIdx.x / 32, epilogue_warp_idx = warp_idx_in_wg;
    for (uint32_t j = threadIdx.x; j < 2 * 32 * 128 / 4; j += 128) reinterpret_cast<uint32_t*>(smem)[j] = 0xdeadbeef;
    __syncthreads();
    constexpr uint32_t ATOM_M = 8, STORE_BLOCK_M_L2 = 32, kSwizzleCDMode = 128, kNumBankGroupBytes = 16;
    #pragma unroll
    for (uint32_t i = 0; i < 4; ++ i) {
        uint32_t row = lane_idx % 8;
        uint32_t col = (epilogue_warp_idx % 2) * 4 + lane_idx / 8;
        const auto smem_ptr = smem + (warp_idx_in_wg / 2) * STORE_BLOCK_M_L2 * kSwizzleCDMode + i * ATOM_M * kSwizzleCDMode +
                              row * (kNumBankGroupBytes * 8) + (col ^ row) * kNumBankGroupBytes;
        const uint32_t* v = vals + (threadIdx.x * 4 + i) * 4;
        ptx::SM90_U32x4_STSM_T<uint32_t>::copy(v[0], v[1], v[2], v[3], smem_ptr);
    }
    __syncthreads();
    for (uint32_t j = threadIdx.x; j < 2 * 32 * 128 / 4; j += 128) out[j] = reinterpret_cast<uint32_t*>(smem)[j];
}

// TMEM: lane L, column C holds L * 65536 + C; `SM100_TMEM_LOAD_16dp256b1x` at columns 8j and 8j | +16 lanes.
__global__ void tmem_ld_ref(uint32_t* out) {
    __shared__ uint32_t holder;
    const uint32_t lane_idx = threadIdx.x % 32, warp_idx = threadIdx.x / 32;
    if (warp_idx == 0) cute::TMEM::Allocator1Sm().allocate(64, &holder);
    cutlass::arch::NamedBarrier::sync(128, 0);
    asm volatile("tcgen05.fence::after_thread_sync;");
    const uint32_t base = holder;
    #pragma unroll
    for (uint32_t c = 0; c < 64; c += 8) {
        uint32_t v[8];
        for (uint32_t k = 0; k < 8; ++ k) v[k] = (warp_idx * 32 + lane_idx) * 65536 + c + k;
        cute::SM100::TMEM::STORE::SM100_TMEM_STORE_32dp32b8x::copy(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], base + ((warp_idx * 32) << 16) + c);
    }
    cutlass::arch::fence_view_async_tmem_store();
    asm volatile("tcgen05.fence::before_thread_sync;");
    cutlass::arch::NamedBarrier::sync(128, 0);
    asm volatile("tcgen05.fence::after_thread_sync;");
    uint32_t values[2][8];
    #pragma unroll
    for (uint32_t j = 0; j < 2; ++ j) {
        const uint32_t tmem_addr = base + j * 8;
        cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr, values[j][0], values[j][1], values[j][2], values[j][3]);
        cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000, values[j][4], values[j][5], values[j][6], values[j][7]);
    }
    cutlass::arch::fence_view_async_tmem_load();
    for (uint32_t j = 0; j < 2; ++ j)
        for (uint32_t k = 0; k < 8; ++ k) out[(threadIdx.x * 2 + j) * 8 + k] = values[j][k];
    asm volatile("tcgen05.fence::before_thread_sync;");
    cutlass::arch::NamedBarrier::sync(128, 0);
    if (warp_idx == 0) cute::TMEM::Allocator1Sm().free(base, 64);
}

void launch_maps(const uint32_t* l1_vals, uint32_t* l1_out, const uint32_t* l2_vals, uint32_t* l2_out, uint32_t* tmem_out) {
    stsm_l1_ref<<<1, 128>>>(l1_vals, l1_out);
    stsm_l2_ref<<<1, 128>>>(l2_vals, l2_out);
    tmem_ld_ref<<<1, 128>>>(tmem_out);
}
