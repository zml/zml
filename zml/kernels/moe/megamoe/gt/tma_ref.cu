// 2-CTA TMA reference: DeepGEMM's `tma::copy` / `copy_nd` (2SM loads, peer-bit barrier mask),
// `ClusterTransactionBarrier` protocol of the MegaMoE TMA warps, `SM90_TMA_STORE_2D` and
// `ptx::tma_load_1d` / `tma_store_1d`, on descriptors encoded with DeepGEMM's parameters
// (`runtime_utils.hpp`: L2 256B, OOB none). Each CTA dumps its shared memory.
#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__
#define DG_IN_CUDA_COMPILATION
#include <cuda.h>
#include <cutlass/arch/barrier.h>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/ptx/tma.cuh>

using namespace deep_gemm;

constexpr uint32_t kSmemA = 0, kSmemB = 16384, kSmemSf = 32768, kSmemBulk = 33792, kSmemBar = 34816, kSmemBytes = 35840;

__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(128, 1)
tma_ref(const __grid_constant__ CUtensorMap map_a, const __grid_constant__ CUtensorMap map_sf,
        const __grid_constant__ CUtensorMap map_b, const __grid_constant__ CUtensorMap map_out,
        const uint8_t* bulk_src, uint8_t* smem_out, uint8_t* store_out, uint8_t* bulk_out) {
    extern __shared__ __align__(1024) uint8_t smem[];
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    auto* bar = reinterpret_cast<Barrier*>(smem + kSmemBar);
    auto* bulk_bar = reinterpret_cast<Barrier*>(smem + kSmemBar + 8);
    const bool is_leader = cute::block_rank_in_cluster() == 0;
    const uint32_t cta = cute::block_rank_in_cluster();
    for (uint32_t i = threadIdx.x; i < kSmemBar / 4; i += 128) reinterpret_cast<uint32_t*>(smem)[i] = 0xdeadbeef;
    if (threadIdx.x == 0) {
        bar->init(2);
        bulk_bar->init(1);
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();
    cute::cluster_sync();
    if (threadIdx.x / 32 == 0) {
        if (cute::elect_one_sync()) {
            const uint32_t m_idx = is_leader ? 0 : 96;
            tma::copy<128, 96, 128, cutlass::float_e4m3_t>(&map_a, bar, reinterpret_cast<cutlass::float_e4m3_t*>(smem + kSmemA), 128, m_idx, 2);
            tma::copy<256, 1, 0>(&map_sf, bar, reinterpret_cast<uint32_t*>(smem + kSmemSf), 0, 1, 2);
            tma::copy_nd<128, 128, 128, cutlass::detail::float_e2m1_unpacksmem_t, 4>(
                &map_b, bar, reinterpret_cast<cutlass::detail::float_e2m1_unpacksmem_t*>(smem + kSmemB), 2, 128, 0, cta, 1);
            if (is_leader) {
                bar->arrive_and_expect_tx(2 * 96 * 128 + 2 * 1024 + 2 * 128 * 128 / 2);
            } else {
                bar->arrive(0u);
            }
            // Non-tensor bulk load of 1024 bytes, then bulk store.
            ptx::tma_load_1d(smem + kSmemBulk, bulk_src + cta * 1024, bulk_bar, 1024);
            ptx::mbarrier_arrive_and_set_tx(bulk_bar, 1024);
            uint32_t phase = 0;
            ptx::mbarrier_wait_and_flip_phase(bulk_bar, phase);
            ptx::tma_store_1d(bulk_out + cta * 1024, smem + kSmemBulk, 1024);
            cute::tma_store_arrive();
            ptx::tma_store_wait<0>();
        }
        __syncwarp();
    }
    if (is_leader) bar->wait(0);
    cute::cluster_sync();
    // TMA store of a 64B-swizzled 32x64 box from the loaded A tile.
    if (threadIdx.x == 0) {
        cute::tma_store_fence();
        cute::SM90_TMA_STORE_2D::copy(&map_out, smem + kSmemA, cta * 64, 0);
        cute::tma_store_arrive();
        ptx::tma_store_wait<0>();
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < kSmemBar; i += 128) smem_out[cta * kSmemBar + i] = smem[i];
}

static CUtensorMap encode(CUtensorMapDataType dtype, uint32_t rank, void* addr, const cuuint64_t* dims, const cuuint64_t* strides,
                          const cuuint32_t* box, CUtensorMapSwizzle swizzle) {
    CUtensorMap map;
    const cuuint32_t elem[5] = {1, 1, 1, 1, 1};
    const auto r = cuTensorMapEncodeTiled(&map, dtype, rank, addr, dims, strides, box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                                          CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) { printf("encode failed %d\n", (int) r); abort(); }
    return map;
}

// acts [192][256] fp8, sf int32 [2][256] (MN contiguous), weights packed fp4 [2][256][128], store out [32][128] u8.
void launch_tma(uint8_t* acts, uint32_t* sf, uint8_t* weights, uint8_t* store_out, uint8_t* bulk_src, uint8_t* smem_out, uint8_t* bulk_out) {
    const cuuint64_t a_dims[2] = {256, 192}, a_strides[1] = {256};
    const cuuint32_t a_box[2] = {128, 96};
    const cuuint64_t sf_dims[2] = {256, 2}, sf_strides[1] = {256 * 4};
    const cuuint32_t sf_box[2] = {256, 1};
    const cuuint64_t b_dims[4] = {256, 128, 2, 2}, b_strides[3] = {128, 128 * 128, 256 * 128};
    const cuuint32_t b_box[4] = {128, 128, 1, 1};
    const cuuint64_t o_dims[2] = {128, 32}, o_strides[1] = {128};
    const cuuint32_t o_box[2] = {64, 32};
    const auto map_a = encode(CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, acts, a_dims, a_strides, a_box, CU_TENSOR_MAP_SWIZZLE_128B);
    const auto map_sf = encode(CU_TENSOR_MAP_DATA_TYPE_INT32, 2, sf, sf_dims, sf_strides, sf_box, CU_TENSOR_MAP_SWIZZLE_NONE);
    const auto map_b = encode(CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, 4, weights, b_dims, b_strides, b_box, CU_TENSOR_MAP_SWIZZLE_128B);
    const auto map_out = encode(CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, store_out, o_dims, o_strides, o_box, CU_TENSOR_MAP_SWIZZLE_64B);
    cudaFuncSetAttribute(tma_ref, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
    tma_ref<<<2, 128, kSmemBytes>>>(map_a, map_sf, map_b, map_out, bulk_src, smem_out, store_out, bulk_out);
}
