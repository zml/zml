// Ground truth for zml/kernels/moe/megamoe/layout.zig, computed by DeepGEMM's own code:
// layout/mega_moe.cuh + scheduler/mega_moe.cuh (vLLM vendored headers), the host heuristics and
// buffer-layout functions of vLLM's DeepGEMM fork (extracted verbatim), and the kernel's
// SharedStorage struct (extracted verbatim). Output: one JSON object on stdout.
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cmath>
#include <string>
#include <vector>
#include <tuple>
#include <algorithm>
#include <utility>
#include <limits>
#define DG_HOST_ASSERT(cond) do { if (!(cond)) { fprintf(stderr, "DG_HOST_ASSERT failed: %s (%s:%d)\n", #cond, __FILE__, __LINE__); abort(); } } while (0)
#define DG_HOST_UNREACHABLE(msg) do { fprintf(stderr, "%s\n", msg); abort(); } while (0)
#define DG_UNIFIED_ASSERT(cond) do { if (!(cond)) { printf("DG_UNIFIED_ASSERT failed: %s\n", #cond); __builtin_trap(); } } while (0)
#define DG_DEVICE_ASSERT(cond) DG_UNIFIED_ASSERT(cond)
#include <cuda_bf16.h>
#include <cutlass/numeric_types.h>
#include <cutlass/float_subbyte.h>
#include <cutlass/arch/barrier.h>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>

namespace deep_gemm {
template <typename T> static T ceil_div(const T& a, const T& b) { return (a + b - 1) / b; }
template <typename T> static constexpr T align(const T& a, const T& b) { return ceil_div(a, b) * b; }
struct SM100ArchSpec {
    static constexpr int smem_capacity = 232448;
    static std::pair<int, int> get_sf_uttcp_aligned_block_sizes(const int& block_m, const int& block_n, const MmaKind& mma_kind) {
        constexpr int num_utccp_aligned_elems = 128;
        switch (mma_kind) {
            case MmaKind::BF16: return {0, 0};
            default: return {align(block_m, num_utccp_aligned_elems), align(block_n, num_utccp_aligned_elems)};
        }
    }
};
static int g_num_sms = 152;
static int get_num_sms_for_mega_moe() { return g_num_sms; }
#include "heur_extract.inc"
namespace mega {

#include "api_extract.inc"
}
}  // namespace deep_gemm

using namespace deep_gemm;

#define OFF(T, f) printf("\"%s\": %zu, ", #f, offsetof(T, f))

static void signals() {
    using T = layout::MegaMoESignals;
    printf("\"signals\": {");
    OFF(T, grid_sync_count); OFF(T, nvl_barrier_counter); OFF(T, nvl_barrier_signals);
    OFF(T, l1_task_count); OFF(T, l2_task_count); OFF(T, shared_l1_task_count); OFF(T, shared_l2_task_count);
    OFF(T, combine_ready_grid_idx); OFF(T, peer_grid_idx); OFF(T, expert_send_count); OFF(T, expert_recv_count);
    OFF(T, expert_recv_count_sum); OFF(T, l1_full_count); OFF(T, l1_empty_count); OFF(T, l2_full_mask);
    OFF(T, l2_empty_count); OFF(T, shared_l2_full_count);
    printf("\"sizeof\": %zu, \"sizeof_token_src_metadata\": %zu, \"sizeof_task_info\": %zu, \"alignof_task_info\": %zu}", sizeof(T), sizeof(layout::TokenSrcMetadata), sizeof(sched::TaskInfo<true>), alignof(sched::TaskInfo<true>));
}

static void buffer_layout(int num_sms, int R, int E, int T_req, int topk, int H, int I, int shared, bool first) {
    g_num_sms = num_sms;
    const int T = mega::get_token_alignment_for_mega_moe() * ceil_div(T_req, mega::get_token_alignment_for_mega_moe());
    const auto info = mega::build_symm_buffer_layout(R, E, T, topk, H, I, "fp8xfp4", "swiglu", shared);
    const auto wsp = layout::Workspace(nullptr, R, E, T, topk, info.num_ring_tokens);
    printf("%s{\"num_sms\": %d, \"num_ranks\": %d, \"num_experts\": %d, \"num_max_tokens_per_rank_req\": %d, \"num_max_tokens_per_rank\": %d, \"num_topk\": %d, \"hidden\": %d, \"intermediate\": %d, \"num_shared_experts\": %d, ",
           first ? "" : ", ", num_sms, R, E, T_req, T, topk, H, I, shared);
    printf("\"layout\": [");
    const auto l = info.to_int_list();
    for (size_t i = 0; i < l.size(); ++ i) printf("%s%lld", i ? ", " : "", (long long) l[i]);
    printf("], \"workspace_bytes\": %llu, \"num_max_pool_tokens\": %u, \"num_shared_l2_pool_blocks\": %u, \"shared_sf_tokens\": %d}",
           (unsigned long long) wsp.get_num_bytes(), wsp.num_max_pool_tokens, wsp.num_shared_l2_pool_blocks,
           layout::get_num_max_shared_sf_tokens(T));
}

static void live_blocks(bool first, int M, int sms, int H, int I) {
    printf("%s[%d, %d, %d, %d, %d, %d]", first ? "" : ", ", M, sms, H, I,
           sched::get_num_max_live_pool_blocks(M, sms, H, I),
           sched::get_num_l1_warmup_waves(M, sms / 2, I * 2 / 256, H / 256));
}

static void config(bool first, int R, int E, int T, int topk, int num_tokens, int H) {
    const auto [cluster, block_m, sbm1, sbm2, block_k, epi] = get_block_config_for_mega_moe(R, E, T, topk, num_tokens, MmaKind::MXFP8FP4);
    const auto [sf_block_m, sf_block_n] = SM100ArchSpec::get_sf_uttcp_aligned_block_sizes(block_m, 128, MmaKind::MXFP8FP4);
    int pull = H;
    while (pull > 8192) pull /= 2;
    const auto [stages, smem] = get_pipeline_config_for_mega_moe(SM100ArchSpec::smem_capacity, E, H, block_m, 128, block_k, pull,
                                                                sbm1, sbm2, sf_block_m, sf_block_n, 32, 4, epi / 32, MmaKind::MXFP8FP4);
    printf("%s[%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d]", first ? "" : ", ", R, E, T, topk, num_tokens, H,
           block_m, sbm1, sbm2, block_k, sf_block_m, pull, stages, smem);
}

template <uint32_t kNumExperts, uint32_t BLOCK_M, uint32_t STORE_BLOCK_M_L1, uint32_t STORE_BLOCK_M_L2, uint32_t kNumStages,
          uint32_t kNumBytesPerPull, typename weight_dtype_t>
static void smem_layout(bool first) {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using a_dtype_t = cutlass::float_e4m3_t;
    using task_info_t = sched::TaskInfo<true>;
    constexpr uint32_t BLOCK_N = 128, BLOCK_K = 128;
    constexpr uint32_t SF_BLOCK_M = math::constexpr_align(BLOCK_M, 128u), SF_BLOCK_N = BLOCK_N;
    constexpr uint32_t kNumDispatchWarps = 4, kNumEpilogueWarps = 8, kNumEpilogueWarpgroups = 2;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / 2, LOAD_BLOCK_N = BLOCK_N;
    constexpr uint32_t kNumEpilogueStages = 2, kNumTMAStoreStages = 2, kSharedMemoryAlignment = 1024;
    constexpr uint32_t kNumScheduleStages = 2;
    constexpr uint32_t L1_OUT_BLOCK_N = BLOCK_N / 2;
    constexpr uint32_t AMAX_REDUCTION_WARP_BUFFER_SIZE = STORE_BLOCK_M_L1 / 2;
#include "smem_extract.inc"
    using S = SharedStorage;
    printf("%s{\"num_experts\": %u, \"block_m\": %u, \"sbm1\": %u, \"sbm2\": %u, \"stages\": %u, \"pull\": %u, ", first ? "" : ", ",
           kNumExperts, BLOCK_M, STORE_BLOCK_M_L1, STORE_BLOCK_M_L2, kNumStages, kNumBytesPerPull);
    OFF(S, expert_token_count); OFF(S, dispatch_send_buffer); OFF(S, smem_d); OFF(S, smem_a); OFF(S, smem_b);
    OFF(S, smem_sfa); OFF(S, smem_sfb); OFF(S, amax_reduction); OFF(S, task_infos); OFF(S, dispatch_barriers);
    OFF(S, full_barriers); OFF(S, empty_barriers); OFF(S, tmem_full_barriers); OFF(S, tmem_empty_barriers);
    OFF(S, combine_barriers); OFF(S, task_info_full_barriers); OFF(S, task_info_empty_barriers); OFF(S, tmem_ptr_in_smem);
    printf("\"sizeof_smem_d\": %zu, \"sizeof\": %zu}", sizeof(S::smem_d), sizeof(S));
}

int main() {
    printf("{");
    signals();
    printf(", \"buffers\": [");
    bool first = true;
    for (int sms: {148, 152, 160}) {
        for (int t: {1, 256, 1024, 4096, 8192, 9600, 16384}) {
            buffer_layout(sms, 2, 384, t, 6, 5120, 2304, 1, first); first = false;
            buffer_layout(sms, 1, 384, t, 6, 5120, 2304, 1, first);
            buffer_layout(sms, 2, 384, t, 6, 5120, 2304, 0, first);
            buffer_layout(sms, 2, 128, t, 3, 5120, 2304, 1, first);
            buffer_layout(sms, 4, 256, t, 8, 7168, 2048, 0, first);
            buffer_layout(sms, 1, 32, t, 4, 1024, 512, 1, first);
        }
    }
    printf("], \"live\": [");
    first = true;
    for (int sms: {148, 152, 160})
        for (int M: {1, 2, 7, 33, 100, 192, 256, 500, 1000, 4000, 9000})
            for (auto [H, I]: std::vector<std::pair<int,int>>{{5120, 2304}, {7168, 2048}, {1024, 512}, {2048, 4096}}) {
                live_blocks(first, M, sms, H, I); first = false;
            }
    printf("], \"configs\": [");
    first = true;
    for (auto [R, E, topk, H]: std::vector<std::tuple<int,int,int,int>>{{2, 384, 6, 5120}, {1, 384, 6, 5120}, {2, 128, 3, 5120}, {4, 256, 8, 7168}, {1, 32, 4, 1024}, {8, 256, 8, 7168}}) {
        for (int n: {0, 1, 3, 16, 64, 128, 192, 256, 320, 321, 384, 512, 768, 769, 1024, 1536, 1807, 1808, 2048, 3072, 3749, 3750, 4096, 5716, 5717, 6144, 7200, 7201, 7695, 7696, 8192, 9600, 11676, 11677, 16384}) {
            config(first, R, E, 9600, topk, n, H); first = false;
        }
    }
    printf("], \"smem\": [");
    using fp4 = cutlass::detail::float_e2m1_unpacksmem_t;
    smem_layout<384, 192, 32, 32, 6, 5120, fp4>(true);
    smem_layout<384, 16, 8, 8, 11, 5120, fp4>(false);
    smem_layout<384, 32, 16, 16, 10, 5120, fp4>(false);
    smem_layout<384, 64, 16, 16, 9, 5120, fp4>(false);
    smem_layout<384, 128, 32, 32, 7, 5120, fp4>(false);
    smem_layout<384, 240, 24, 8, 6, 5120, fp4>(false);
    smem_layout<128, 192, 32, 32, 6, 5120, fp4>(false);
    smem_layout<32, 64, 16, 16, 10, 1024, fp4>(false);
    smem_layout<256, 240, 24, 8, 5, 7168, fp4>(false);
    smem_layout<384, 192, 32, 32, 6, 5120, cutlass::float_e4m3_t>(false);
    printf("]}\n");
    return 0;
}
