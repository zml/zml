#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_atomic.h>

// Helper to convert f4e2m1 (nvfp4) to bf16
// f4e2m1 format: 1 sign bit, 2 exponent bits, 1 mantissa bit
// Values: 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, and negatives
__device__ __forceinline__ __nv_bfloat16 dequantize_f4e2m1_to_bf16(uint8_t quantized, uint8_t scale_f8e8m0) {
    // Lookup table for f4e2m1 values (16 possible values)
    constexpr float f4e2m1_values[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    
    // Convert f8e8m0 scale to float
    // f8e8m0: 8 exponent bits, 0 mantissa bits (just exponent)
    float scale = __exp2f((float)((int)scale_f8e8m0 - 127));
    
    // Extract f4e2m1 value (low 4 bits)
    uint8_t f4_val = quantized & 0x0F;
    float dequantized = f4e2m1_values[f4_val] * scale;
    
    return __float2bfloat16(dequantized);
}

// Quick GELU activation
__device__ __forceinline__ __nv_bfloat16 quick_gelu(__nv_bfloat16 x) {
    const float xf = __bfloat162float(x);
    const float gelu = xf * 0.5f * (1.0f + tanhf(0.79788456f * (xf + 0.044715f * xf * xf * xf)));
    return __float2bfloat16(gelu);
}

// Kernel to extract k-th column from 2D array and compute restore indices
__global__ void extract_topk_and_compute_restore_kernel(
    const int* __restrict__ expert_indices_2d,      // [seq_len, top_k]
    const float* __restrict__ routing_scores_2d,  // [seq_len, top_k]
    const int* __restrict__ token_mask,           // [seq_len] or nullptr
    int* __restrict__ expert_indices_1d,          // [seq_len] output
    float* __restrict__ routing_scores_1d,        // [seq_len] output
    int* __restrict__ restore_indices,            // [seq_len] output - argsort of sorted indices
    int seq_len,
    int top_k,
    int k_rank
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len) return;
    
    if (token_mask && token_mask[idx] == 0) {
        expert_indices_1d[idx] = -1;  // Invalid token
        routing_scores_1d[idx] = 0.0f;
        return;
    }
    
    expert_indices_1d[idx] = expert_indices_2d[idx * top_k + k_rank];
    routing_scores_1d[idx] = routing_scores_2d[idx * top_k + k_rank];
    restore_indices[idx] = idx;  // Will be updated by sort kernel
}

// Kernel to sort tokens by expert and compute offsets
// Uses CUB for efficient sorting
__global__ void sort_tokens_by_expert_kernel(
    const int* __restrict__ expert_indices,        // [seq_len] - expert ID for each token
    const int* __restrict__ token_mask,           // [seq_len] or nullptr
    int* __restrict__ sorted_token_indices,       // [seq_len] - output: sorted token indices
    int* __restrict__ expert_offsets,             // [num_experts + 1] - output: cumulative offsets
    int seq_len,
    int num_experts
) {
    // Use shared memory for counting
    extern __shared__ int shared_counts[];
    int* expert_counts = shared_counts;
    
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    // Initialize counts
    if (tid < num_experts) {
        expert_counts[tid] = 0;
    }
    __syncthreads();
    
    // Count tokens per expert (only valid tokens)
    for (int i = tid; i < seq_len; i += num_threads) {
        if (token_mask && token_mask[i] == 0) continue;
        int expert_id = expert_indices[i];
        if (expert_id >= 0 && expert_id < num_experts) {
            atomicAdd(&expert_counts[expert_id], 1);
        }
    }
    __syncthreads();
    
    // Compute cumulative offsets
    if (tid == 0) {
        expert_offsets[0] = 0;
        for (int i = 0; i < num_experts; i++) {
            expert_offsets[i + 1] = expert_offsets[i] + expert_counts[i];
        }
    }
    __syncthreads();
    
    // Reset counts for second pass
    if (tid < num_experts) {
        expert_counts[tid] = 0;
    }
    __syncthreads();
    
    // Scatter tokens to sorted positions
    for (int i = tid; i < seq_len; i += num_threads) {
        if (token_mask && token_mask[i] == 0) continue;
        int expert_id = expert_indices[i];
        if (expert_id >= 0 && expert_id < num_experts) {
            int pos = expert_offsets[expert_id] + atomicAdd(&expert_counts[expert_id], 1);
            sorted_token_indices[pos] = i;
        }
    }
}

// Kernel to compute restore indices (inverse permutation)
__global__ void compute_restore_indices_kernel(
    const int* __restrict__ sorted_token_indices, // [seq_len] - sorted indices
    int* __restrict__ restore_indices,            // [seq_len] - output: restore[i] = position of token i in sorted array
    int seq_len
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len) return;
    
    int original_token = sorted_token_indices[idx];
    restore_indices[original_token] = idx;
}

// Main fused MoE kernel for one expert
// Processes all tokens assigned to this expert at current top-k rank
__global__ void fused_moe_expert_kernel(
    // Inputs
    const __nv_bfloat16* __restrict__ input,              // [seq_len, hidden_dim]
    const uint8_t* __restrict__ gate_up_blocks,           // [num_experts, ffn_dim*2, hidden_dim] quantized
    const uint8_t* __restrict__ gate_up_scales,           // [num_experts, ffn_dim*2, hidden_dim] scales
    const __nv_bfloat16* __restrict__ gate_up_bias,      // [num_experts, ffn_dim*2] or nullptr
    const uint8_t* __restrict__ down_blocks,              // [num_experts, hidden_dim, ffn_dim] quantized
    const uint8_t* __restrict__ down_scales,              // [num_experts, hidden_dim, ffn_dim] scales
    const __nv_bfloat16* __restrict__ down_bias,          // [num_experts, hidden_dim] or nullptr
    const int* __restrict__ sorted_token_indices,         // [seq_len] - tokens sorted by expert
    const int* __restrict__ expert_offsets,               // [num_experts + 1] - cumulative offsets
    const int* __restrict__ restore_indices,             // [seq_len] - restore original order
    const float* __restrict__ routing_scores,              // [seq_len] - routing scores for original tokens
    
    // Outputs (accumulated)
    __nv_bfloat16* __restrict__ output,                   // [seq_len, hidden_dim]
    
    // Parameters
    int seq_len,
    int num_experts,
    int hidden_dim,
    int ffn_dim,
    int block_size,
    int expert_id
) {
    const int start_token = expert_offsets[expert_id];
    const int end_token = expert_offsets[expert_id + 1];
    const int num_tokens = end_token - start_token;
    
    if (num_tokens == 0) return;
    
    // Each thread block processes one token
    const int token_local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_feature = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (token_local_idx >= num_tokens || out_feature >= hidden_dim) return;
    
    const int original_token_idx = sorted_token_indices[start_token + token_local_idx];
    const int restore_idx = restore_indices[original_token_idx];
    const float score = routing_scores[original_token_idx];
    
    const int original_token_idx = sorted_token_indices[start_token + token_local_idx];
    const int restore_idx = restore_indices[original_token_idx];
    const float score = routing_scores[original_token_idx];
    
    // Use shared memory to store gate-up results for this token
    extern __shared__ __nv_bfloat16 shared_gate_up[];
    __nv_bfloat16* gate_up_results = shared_gate_up;  // [ffn_dim] per token
    
    // Step 1: Gate-Up projection for this token
    // Each thread processes one ffn_dim feature (with gate and up interleaved)
    const int feat = threadIdx.y;
    
    if (feat < ffn_dim) {
        float gate_acc = 0.0f;
        float up_acc = 0.0f;
        
        for (int in_feat = threadIdx.x; in_feat < hidden_dim; in_feat += blockDim.x) {
            // Gate weight at position 2*feat (interleaved format)
            const int gate_idx = expert_id * ffn_dim * 2 * hidden_dim + (feat * 2) * hidden_dim + in_feat;
            const int gate_weight_block = gate_idx / block_size;
            const int gate_scale_idx = gate_idx;
            uint8_t gate_quantized = gate_up_blocks[gate_weight_block];
            uint8_t gate_scale = gate_up_scales[gate_scale_idx];
            __nv_bfloat16 gate_weight = dequantize_f4e2m1_to_bf16(gate_quantized, gate_scale);
            
            // Up weight at position 2*feat+1
            const int up_idx = expert_id * ffn_dim * 2 * hidden_dim + (feat * 2 + 1) * hidden_dim + in_feat;
            const int up_weight_block = up_idx / block_size;
            const int up_scale_idx = up_idx;
            uint8_t up_quantized = gate_up_blocks[up_weight_block];
            uint8_t up_scale = gate_up_scales[up_scale_idx];
            __nv_bfloat16 up_weight = dequantize_f4e2m1_to_bf16(up_quantized, up_scale);
            
            float input_val = __bfloat162float(input[original_token_idx * hidden_dim + in_feat]);
            gate_acc += input_val * __bfloat162float(gate_weight);
            up_acc += input_val * __bfloat162float(up_weight);
        }
        
        // Reduce across threads in x dimension
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            gate_acc += __shfl_down_sync(0xFFFFFFFF, gate_acc, offset);
            up_acc += __shfl_down_sync(0xFFFFFFFF, up_acc, offset);
        }
        
        if (threadIdx.x == 0) {
            // Add bias (also interleaved)
            if (gate_up_bias) {
                gate_acc += __bfloat162float(gate_up_bias[expert_id * ffn_dim * 2 + feat * 2]);
                up_acc += __bfloat162float(gate_up_bias[expert_id * ffn_dim * 2 + feat * 2 + 1]);
            }
            
            // Clamp
            gate_acc = fminf(gate_acc, 7.0f);
            up_acc = fmaxf(fminf(up_acc, 7.0f), -7.0f);
            
            // Apply QuickGELU and multiply
            __nv_bfloat16 gate = __float2bfloat16(gate_acc);
            __nv_bfloat16 up = __float2bfloat16(up_acc);
            __nv_bfloat16 activated = quick_gelu(gate);
            activated = __float2bfloat16(__bfloat162float(activated) * (__bfloat162float(up) + 1.0f));
            
            // Store in shared memory
            gate_up_results[feat] = activated;
        }
    }
    __syncthreads();
    
    // Step 2: Down projection
    // Each thread processes one output feature
    if (out_feature < hidden_dim) {
        float down_acc = 0.0f;
        
        for (int in_feat = threadIdx.x; in_feat < ffn_dim; in_feat += blockDim.x) {
            const int down_weight_block = (expert_id * hidden_dim * ffn_dim + out_feature * ffn_dim + in_feat) / block_size;
            const int down_scale_idx = expert_id * hidden_dim * ffn_dim + out_feature * ffn_dim + in_feat;
            uint8_t down_quantized = down_blocks[down_weight_block];
            uint8_t down_scale = down_scales[down_scale_idx];
            __nv_bfloat16 down_weight = dequantize_f4e2m1_to_bf16(down_quantized, down_scale);
            
            down_acc += __bfloat162float(gate_up_results[in_feat]) * __bfloat162float(down_weight);
        }
        
        // Reduce across threads
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            down_acc += __shfl_down_sync(0xFFFFFFFF, down_acc, offset);
        }
        
        if (threadIdx.x == 0) {
            if (down_bias) {
                down_acc += __bfloat162float(down_bias[expert_id * hidden_dim + out_feature]);
            }
            
            // Multiply by routing score and accumulate to output
            down_acc *= score;
            __nv_bfloat16 result = __float2bfloat16(down_acc);
            
            // Atomic add to output (multiple experts can contribute to same token)
            // Use atomicAdd on unsigned short (bf16 is 16 bits)
            unsigned short* output_ushort = (unsigned short*)&output[restore_idx * hidden_dim + out_feature];
            unsigned short result_ushort = __bfloat162ushort(result);
            
            // Atomic add using compare-and-swap loop for bf16
            unsigned short old_val, new_val;
            do {
                old_val = *output_ushort;
                float old_float = __ushort2bfloat16(old_val);
                float new_float = old_float + __bfloat162float(result);
                new_val = __bfloat162ushort(__float2bfloat16(new_float));
            } while (atomicCAS(output_ushort, old_val, new_val) != old_val);
        }
    }
}

// Host function to launch the complete fused MoE kernel
extern "C" int fused_moe_kernel_launch(
    void* stream,
    // Inputs (all on device)
    const void* input,                    // [seq_len, hidden_dim] bf16
    const void* gate_up_blocks,           // [num_experts, ffn_dim*2, hidden_dim] quantized f4e2m1
    const void* gate_up_scales,           // [num_experts, ffn_dim*2, hidden_dim] f8e8m0
    const void* gate_up_bias,            // [num_experts, ffn_dim*2] bf16 or nullptr
    const void* down_blocks,              // [num_experts, hidden_dim, ffn_dim] quantized f4e2m1
    const void* down_scales,              // [num_experts, hidden_dim, ffn_dim] f8e8m0
    const void* down_bias,                // [num_experts, hidden_dim] bf16 or nullptr
    const int* expert_indices,            // [seq_len, top_k] - expert IDs
    const float* routing_scores,          // [seq_len, top_k] - routing scores
    const int* token_mask,                // [seq_len] - 1 if valid, 0 otherwise, or nullptr
    void* workspace,                      // Device workspace for intermediate arrays
    void* output,                         // [seq_len, hidden_dim] bf16 (initialized to zero)
    int seq_len,
    int num_experts,
    int top_k,
    int hidden_dim,
    int ffn_dim,
    int block_size
) {
    cudaStream_t cuda_stream = (cudaStream_t)stream;
    cudaError_t err;
    
    // Workspace layout (all on device):
    // - sorted_token_indices: [seq_len] * top_k
    // - expert_offsets: [num_experts + 1] * top_k
    // - restore_indices: [seq_len] * top_k
    // - expert_indices_1d: [seq_len]
    // - routing_scores_1d: [seq_len]
    
    int* sorted_token_indices_base = (int*)workspace;
    int* expert_offsets_base = sorted_token_indices_base + seq_len * top_k;
    int* restore_indices_base = expert_offsets_base + (num_experts + 1) * top_k;
    int* expert_indices_1d = restore_indices_base + seq_len * top_k;
    float* routing_scores_1d = (float*)(expert_indices_1d + seq_len);
    
    // Process each top-k rank
    for (int k = 0; k < top_k; k++) {
        int* sorted_token_indices = sorted_token_indices_base + k * seq_len;
        int* expert_offsets = expert_offsets_base + k * (num_experts + 1);
        int* restore_indices = restore_indices_base + k * seq_len;
        
        // Step 1: Extract k-th column from expert_indices and routing_scores
        dim3 extract_grid((seq_len + 255) / 256);
        dim3 extract_block(256);
        extract_topk_and_compute_restore_kernel<<<extract_grid, extract_block, 0, cuda_stream>>>(
            expert_indices,
            routing_scores,
            token_mask,
            expert_indices_1d,
            routing_scores_1d,
            restore_indices,  // Will be properly computed later
            seq_len,
            top_k,
            k
        );
        
        // Step 2: Sort tokens by expert
        dim3 sort_grid(1);
        dim3 sort_block(256);
        size_t sort_shared = num_experts * sizeof(int);
        sort_tokens_by_expert_kernel<<<sort_grid, sort_block, sort_shared, cuda_stream>>>(
            expert_indices_1d,
            token_mask,
            sorted_token_indices,
            expert_offsets,
            seq_len,
            num_experts
        );
        
        // Step 3: Compute restore indices (inverse permutation)
        dim3 restore_grid((seq_len + 255) / 256);
        dim3 restore_block(256);
        compute_restore_indices_kernel<<<restore_grid, restore_block, 0, cuda_stream>>>(
            sorted_token_indices,
            restore_indices,
            seq_len
        );
        
        // Step 4: Process each expert
        // Grid: (max_tokens_per_expert, hidden_dim/32)
        // Block: (32, 32) - 32 threads for input features, 32 for output features
        dim3 moe_block(32, 32);
        size_t moe_shared = ffn_dim * sizeof(__nv_bfloat16);  // For gate_up results per token
        
        for (int expert_id = 0; expert_id < num_experts; expert_id++) {
            int num_tokens_expert = expert_offsets[expert_id + 1] - expert_offsets[expert_id];
            if (num_tokens_expert == 0) continue;
            
            dim3 moe_grid((num_tokens_expert + 31) / 32, (hidden_dim + 31) / 32);
            
            fused_moe_expert_kernel<<<moe_grid, moe_block, moe_shared, cuda_stream>>>(
                (const __nv_bfloat16*)input,
                (const uint8_t*)gate_up_blocks,
                (const uint8_t*)gate_up_scales,
                (const __nv_bfloat16*)gate_up_bias,
                (const uint8_t*)down_blocks,
                (const uint8_t*)down_scales,
                (const __nv_bfloat16*)down_bias,
                sorted_token_indices,
                expert_offsets,
                restore_indices,
                routing_scores_1d,
                (__nv_bfloat16*)output,
                seq_len,
                num_experts,
                hidden_dim,
                ffn_dim,
                block_size,
                expert_id
            );
        }
    }
    
    err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : (int)err;
}
