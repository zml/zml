// Native launcher for AITER's gfx942 fused block-scaled FP8 MoE kernel.
//
// The code object is shipped by ROCm/AITER under the MIT license. Keeping the
// launcher behind a plain C ABI lets XLA FFI pass PJRT's current HIP stream
// without introducing a Torch dependency.

#include <cstddef>
#include <cstdint>
#include <array>
#include <mutex>
#include <string>

#include <hip/hip_runtime_api.h>

namespace {

struct P3
{
    uint32_t p0;
    uint32_t p1;
    uint32_t p2;
};

struct P2
{
    uint32_t p0;
    uint32_t p1;
};

struct __attribute__((packed)) KernelArgs
{
    void* output;
    P2 pad0;
    const void* input;
    P2 pad1;
    const void* gate_up;
    P2 pad2;
    const void* num_valid_ids;
    P2 pad3;
    const void* down;
    P2 pad4;
    const void* input_scale;
    P2 pad5;
    const void* gate_up_scale;
    P2 pad6;
    const void* down_scale;
    P2 pad7;
    const void* smooth_scale;
    P2 pad8;
    const void* sorted_token_ids;
    P2 pad9;
    const void* sorted_weights;
    P2 pad10;
    const void* sorted_expert_ids;
    P2 pad11;
    uint32_t hidden_size;
    P3 pad12;
    uint32_t intermediate_size;
    P3 pad13;
    uint32_t tokens;
    P3 pad14;
    uint32_t experts;
    P3 pad15;
    uint32_t input_stride;
    P3 pad16;
    uint32_t gate_up_stride;
    P3 pad17;
    uint32_t down_stride;
    P3 pad18;
    uint32_t output_stride;
    P3 pad19;
    uint32_t expert_gate_up_stride;
    P3 pad20;
    uint32_t expert_down_stride;
    P3 pad21;
    uint32_t expert_gate_up_scale_stride;
    P3 pad22;
    uint32_t expert_down_scale_stride;
    P3 pad23;
    uint32_t expert_smooth_scale_stride;
    P3 pad24;
    uint32_t topk;
    P3 pad25;
    uint32_t total_thread_groups;
    P3 pad26;
    uint32_t persistent_denominator;
    P3 pad27;
};

static_assert(sizeof(KernelArgs) == 448);

struct DeviceKernel
{
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;
};

std::array<DeviceKernel, 128> device_kernels{};
std::mutex device_kernels_mutex;
std::string code_object_path;

constexpr char kernel_name[] =
    "_ZN5aiter49fmoe_bf16_blockscaleFp8_g1u1_novs_silu_1tg_32x256E";

hipFunction_t functionForCurrentDevice()
{
    int device = -1;
    if(hipGetDevice(&device) != hipSuccess || device < 0 ||
       static_cast<std::size_t>(device) >= device_kernels.size())
        return nullptr;

    std::lock_guard<std::mutex> lock(device_kernels_mutex);
    auto& kernel = device_kernels[static_cast<std::size_t>(device)];
    if(kernel.function != nullptr) return kernel.function;
    if(code_object_path.empty()) return nullptr;
    if(hipModuleLoad(&kernel.module, code_object_path.c_str()) != hipSuccess) return nullptr;
    if(hipModuleGetFunction(&kernel.function, kernel.module, kernel_name) != hipSuccess)
    {
        static_cast<void>(hipModuleUnload(kernel.module));
        kernel = {};
        return nullptr;
    }
    return kernel.function;
}

} // namespace

extern "C" int zml_aiter_moe_init(const char* code_object_path)
{
    if(code_object_path == nullptr || code_object_path[0] == '\0') return 1;
    std::lock_guard<std::mutex> lock(device_kernels_mutex);
    ::code_object_path = code_object_path;
    return 0;
}

extern "C" int zml_aiter_moe_a8w8_blockscale(
    void* stream,
    const int32_t* sorted_token_ids,
    const int32_t* sorted_expert_ids,
    const int32_t* num_valid_ids,
    const float* sorted_weights,
    const void* hidden,
    const void* gate_up,
    const void* down,
    const float* input_scale,
    const float* gate_up_scale,
    const float* down_scale,
    void* output,
    int64_t tokens,
    int64_t experts,
    int64_t topk,
    int64_t sorted_blocks,
    int64_t hidden_size,
    int64_t intermediate_size)
{
    const auto function = functionForCurrentDevice();
    if(function == nullptr) return 1;
    if(tokens <= 0 || experts <= 0 || topk <= 0 || sorted_blocks <= 0 ||
       hidden_size <= 0 || intermediate_size <= 0 || hidden_size % 128 != 0 ||
       intermediate_size % 256 != 0)
        return 2;

    const auto hip_stream = reinterpret_cast<hipStream_t>(stream);
    if(hipMemsetAsync(
           output,
           0,
           static_cast<std::size_t>(tokens * hidden_size * sizeof(uint16_t)),
           hip_stream) != hipSuccess)
        return 3;

    KernelArgs args{};
    args.output = output;
    args.input = hidden;
    args.gate_up = gate_up;
    args.num_valid_ids = num_valid_ids;
    args.down = down;
    args.input_scale = input_scale;
    args.gate_up_scale = gate_up_scale;
    args.down_scale = down_scale;
    args.sorted_token_ids = sorted_token_ids;
    args.sorted_weights = sorted_weights;
    args.sorted_expert_ids = sorted_expert_ids;
    args.hidden_size = static_cast<uint32_t>(hidden_size);
    args.intermediate_size = static_cast<uint32_t>(intermediate_size);
    args.tokens = static_cast<uint32_t>(tokens);
    args.experts = static_cast<uint32_t>(experts);
    args.input_stride = static_cast<uint32_t>(hidden_size);
    args.gate_up_stride = static_cast<uint32_t>(hidden_size);
    args.down_stride = static_cast<uint32_t>(intermediate_size);
    args.output_stride = static_cast<uint32_t>(hidden_size * sizeof(uint16_t));
    args.expert_gate_up_stride = static_cast<uint32_t>(2 * intermediate_size * hidden_size);
    args.expert_down_stride = static_cast<uint32_t>(hidden_size * intermediate_size);
    args.expert_gate_up_scale_stride = static_cast<uint32_t>(
        2 * intermediate_size / 128 * (hidden_size / 128) * sizeof(float));
    args.expert_down_scale_stride = static_cast<uint32_t>(
        hidden_size / 128 * (intermediate_size / 128) * sizeof(float));
    args.expert_smooth_scale_stride = static_cast<uint32_t>(intermediate_size * sizeof(float));
    args.topk = static_cast<uint32_t>(topk);
    args.persistent_denominator = static_cast<uint32_t>(intermediate_size / 256);

    std::size_t arg_size = sizeof(args);
    void* config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        &args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        &arg_size,
        HIP_LAUNCH_PARAM_END,
    };
    const auto launch_status = hipModuleLaunchKernel(
        function,
        static_cast<unsigned int>(intermediate_size / 256),
        static_cast<unsigned int>(sorted_blocks),
        1,
        256,
        1,
        1,
        0,
        hip_stream,
        nullptr,
        config);
    return launch_status == hipSuccess ? 0 : 4;
}
