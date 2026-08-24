// Native gfx942 block-scaled FP8 kernels built with ROCm Composable Kernel.
// The ABI deliberately contains no Torch/AITER types so it can be called from
// XLA FFI with the execution stream supplied by PJRT.

#include <array>
#include <cstdint>

#include <hip/hip_runtime_api.h>

#include "ck/ck.hpp"
#include "ck/stream_config.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_ab_scale.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/blkgemmpipe_scheduler.hpp"

namespace {

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F8 = ck::f8_t;
using BF16 = ck::bhalf_t;
using F32 = float;
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using GemmSpecialization = ck::tensor_operation::device::GemmSpecialization;

template <GemmSpecialization Specialization>
using DeviceGemm = ck::tensor_operation::device::DeviceGemmMultiD_ABScale_Xdl_CShuffle_V3<
    Row,
    Col,
    ck::Tuple<>,
    Row,
    F8,
    F32,
    F8,
    F32,
    ck::Tuple<>,
    BF16,
    F32,
    F32,
    PassThrough,
    PassThrough,
    PassThrough,
    Specialization,
    256,        // block size
    1, 128, 128, // scale blocks M/N/K
    16, 128, 256,
    16, 16,
    16, 16,
    1, 2,
    S<16, 16, 1>,
    S<1, 0, 2>,
    S<1, 0, 2>,
    2, 16, 16, 0,
    S<16, 16, 1>,
    S<1, 0, 2>,
    S<1, 0, 2>,
    2, 16, 16, 0,
    1, 2,
    S<1, 16, 1, 16>,
    S<8>,
    ck::BlockGemmPipelineScheduler::Intrawave,
    ck::BlockGemmPipelineVersion::v1,
    F8>;

template <GemmSpecialization Specialization>
int runGemm(
    hipStream_t stream,
    const void* a,
    const void* b,
    const float* a_scale,
    const float* b_scale,
    void* c,
    int64_t m,
    int64_t n,
    int64_t k)
{
    using Op = DeviceGemm<Specialization>;
    constexpr ck::index_t NumDTensor = ck::Tuple<>::Size();

    Op op;
    auto argument = op.MakeArgument(
        a,
        b,
        std::array<const void*, NumDTensor>{},
        reinterpret_cast<BF16*>(c),
        m,
        n,
        k,
        k,
        k,
        std::array<ck::index_t, NumDTensor>{},
        n,
        a_scale,
        b_scale,
        PassThrough{},
        PassThrough{},
        PassThrough{});

    if(!op.IsSupportedArgument(argument)) return 1;
    auto invoker = op.MakeInvoker();
    invoker.Run(argument, StreamConfig{stream});
    return 0;
}

} // namespace

extern "C" int zml_ck_gemm_a8w8_blockscale(
    void* stream,
    const void* a,
    const void* b,
    const float* a_scale,
    const float* b_scale,
    void* c,
    int64_t m,
    int64_t n,
    int64_t k)
{
    const bool mp = m % 16 != 0;
    const bool np = n % 128 != 0;
    const bool kp = k % 256 != 0;
    const unsigned padding = unsigned(mp) | (unsigned(np) << 1) | (unsigned(kp) << 2);
    const auto hip_stream = reinterpret_cast<hipStream_t>(stream);

    switch(padding)
    {
    case 0: return runGemm<GemmSpecialization::Default>(hip_stream, a, b, a_scale, b_scale, c, m, n, k);
    case 1: return runGemm<GemmSpecialization::MPadding>(hip_stream, a, b, a_scale, b_scale, c, m, n, k);
    case 2: return runGemm<GemmSpecialization::NPadding>(hip_stream, a, b, a_scale, b_scale, c, m, n, k);
    case 3: return runGemm<GemmSpecialization::MNPadding>(hip_stream, a, b, a_scale, b_scale, c, m, n, k);
    case 4: return runGemm<GemmSpecialization::KPadding>(hip_stream, a, b, a_scale, b_scale, c, m, n, k);
    case 5: return runGemm<GemmSpecialization::MKPadding>(hip_stream, a, b, a_scale, b_scale, c, m, n, k);
    case 6: return runGemm<GemmSpecialization::NKPadding>(hip_stream, a, b, a_scale, b_scale, c, m, n, k);
    case 7: return runGemm<GemmSpecialization::MNKPadding>(hip_stream, a, b, a_scale, b_scale, c, m, n, k);
    default: return 2;
    }
}
