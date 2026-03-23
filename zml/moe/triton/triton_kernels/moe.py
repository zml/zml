import triton
import triton.language as tl


@triton.jit
def write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_bias_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N_ptr,
    K_ptr,
    EM_ptr,
    num_valid_tokens_ptr,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am_ptr,
    stride_ak,
    stride_be_ptr,
    stride_bk,
    stride_bn_ptr,
    stride_cm_ptr,
    stride_cn,
    stride_asm_ptr,
    stride_ask_ptr,
    stride_bse_ptr,
    stride_bsk_ptr,
    stride_bsn_ptr,
    stride_bbe_ptr,  # bias expert stride
    stride_bbn_ptr,  # bias N stride
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    naive_block_assignment: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    out_ptr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    - naive_block_assignment: A boolean flag indicating whether to use naive
        token wise block assignment. If True, each block corresponds to a
        single token.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    n_raw = tl.load(N_ptr).to(tl.int64)
    k_raw = tl.load(K_ptr).to(tl.int64)
    em_raw = tl.load(EM_ptr).to(tl.int64)
    num_valid_tokens = tl.load(num_valid_tokens_ptr).to(tl.int64)

    stride_am_raw = tl.load(stride_am_ptr).to(tl.int64)
    stride_ak_raw = stride_ak
    stride_be_raw = tl.load(stride_be_ptr).to(tl.int64)
    stride_bk_raw = stride_bk
    stride_bn_raw = tl.load(stride_bn_ptr).to(tl.int64)
    stride_cm_raw = tl.load(stride_cm_ptr).to(tl.int64)
    stride_cn_raw = stride_cn
    stride_asm_raw = tl.load(stride_asm_ptr).to(tl.int64)
    stride_ask_raw = tl.load(stride_ask_ptr).to(tl.int64)
    stride_bse_raw = tl.load(stride_bse_ptr).to(tl.int64)
    stride_bsk_raw = tl.load(stride_bsk_ptr).to(tl.int64)
    stride_bsn_raw = tl.load(stride_bsn_ptr).to(tl.int64)
    stride_bbe_raw = tl.load(stride_bbe_ptr).to(tl.int64)
    stride_bbn_raw = tl.load(stride_bbn_ptr).to(tl.int64)

    n_block = (n_raw // 16) * 16
    k_block = (k_raw // 16) * 16
    em_block = (em_raw // 16) * 16
    stride_am_block = (stride_am_raw // 16) * 16
    stride_ak_block = stride_ak_raw
    stride_be_block = (stride_be_raw // 16) * 16
    stride_bk_block = stride_bk_raw
    stride_bn_block = (stride_bn_raw // 16) * 16
    stride_cm_block = (stride_cm_raw // 16) * 16
    stride_cn_block = stride_cn_raw
    stride_asm_block = (stride_asm_raw // 16) * 16
    stride_ask_block = (stride_ask_raw // 16) * 16
    stride_bse_block = (stride_bse_raw // 16) * 16
    stride_bsk_block = (stride_bsk_raw // 16) * 16
    stride_bsn_block = (stride_bsn_raw // 16) * 16
    stride_bbe_block = (stride_bbe_raw // 16) * 16
    stride_bbn_block = (stride_bbn_raw // 16) * 16

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(em_block, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(n_block, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    if not naive_block_assignment:
        offs_token_id = pid_m * BLOCK_SIZE_M + offs
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    else:
        offs_token = tl.where(
            offs == 0,
            pid_m,  # first element = pid_m
            num_valid_tokens,  # remaining elements = constant
        )
    # Cast to int64 to prevent overflow in stride*offset products
    # (e.g. stride_cm * offs_token can exceed int32 for large token counts)
    offs_token = offs_token.to(tl.int64)

    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        write_zeros_to_output(
            c_ptr,
            stride_cm_block,
            stride_cn_block,
            pid_n,
            n_block,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (
        pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    ) % n_block
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am_block
        + offs_k[None, :] * stride_ak_block
    )

    b_ptrs = (
        b_ptr
        + off_experts * stride_be_block
        + (offs_k[:, None] * stride_bk_block + offs_bn[None, :] * stride_bn_block)
    )
    if use_int8_w8a16:
        b_scale_ptrs = (
            b_scale_ptr
            + off_experts * stride_bse_block
            + offs_bn[None, :] * stride_bsn_block
        )
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8 or use_int8_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm_block
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse_block + offs_bsn * stride_bsn_block
            )
        # channel-wise
        elif per_channel_quant:
            b_scale_ptrs = (
                b_scale_ptr
                + off_experts * stride_bse_block
                + offs_bn[None, :] * stride_bsn_block
            )
            b_scale = tl.load(b_scale_ptrs)
            # Load per-token scale for activations
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm_block
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)
    if HAS_BIAS:
        # bias shape: [num_experts, N]
        bias_ptrs = (
            b_bias_ptr + off_experts * stride_bbe_block + offs_bn * stride_bbn_block
        )
        bias = tl.load(bias_ptrs, mask=(offs_bn < n_block), other=0.0)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(k_block, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None]
            & (offs_k[None, :] < k_block - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=offs_k[:, None] < k_block - k * BLOCK_SIZE_K,
            other=0.0,
        )
        # We accumulate along the K dimension.
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_SIZE_K
                offs_ks = k_start // group_k
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask_block,
                    mask=token_mask,
                    other=0.0,
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk_block)

                accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            else:
                if use_fp8_w8a8:
                    # acc used to enable fp8_fast_accum
                    accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak_block
        b_ptrs += BLOCK_SIZE_K * stride_bk_block

    # Dequantization for supported quantization schemes:
    #   - int8_w8a16
    #   - fp8_w8a8
    #   - int8_w8a8
    # Accumulator and scalings are in float32 to preserve numerical accuracy.
    if use_int8_w8a16:
        accumulator = accumulator * b_scale
    elif (use_fp8_w8a8 or use_int8_w8a8) and not (group_k > 0 and group_n > 0):
        accumulator = accumulator * a_scale * b_scale

    # Bias addition:
    # Bias must be applied after dequantization:
    #   - Since bias is typically not quantized
    #   - Bias should not be scaled by quantization factors
    if HAS_BIAS:
        accumulator += bias[None, :]

    # Router (MoE) weight multiplication:
    # This multiplication MUST be performed in float32 before any precision
    # conversion to ensure numerical stability, which is especially critical
    # on ROCm platforms.
    # if MUL_ROUTED_WEIGHT:
    #     moe_weight = tl.load(
    #         topk_weights_ptr + offs_token,
    #         mask=token_mask,
    #         other=0,
    #     )
    #     accumulator *= moe_weight[:, None]

    # Final precision conversion:
    # Cast once at the end to the desired compute/output dtype.
    accumulator = accumulator.to(compute_type)

    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = (
        c_ptr + stride_cm_block * offs_token[:, None] + stride_cn_block * offs_cn[None, :]
    )
    c_mask = token_mask[:, None] & (offs_cn[None, :] < n_block)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def moe_align_block_size_kernel(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_pad_ptr,
    cumsum_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    NUMEL: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    PADDED_NUM_EXPERTS: tl.constexpr,
    MAX_NUM_TOKENS_PADDED: tl.constexpr,
    MAX_NUM_M_BLOCKS: tl.constexpr,
    HIST_BLOCK: tl.constexpr,
    out0_ptr,
    out1_ptr,
    out2_ptr,
    out3_ptr,
):
    pid = tl.program_id(axis=0)

    fill_offs = tl.arange(0, HIST_BLOCK)
    if pid == 1:
        for start in range(0, MAX_NUM_TOKENS_PADDED, HIST_BLOCK):
            offs = start + fill_offs
            tl.store(sorted_token_ids_ptr + offs, NUMEL, mask=offs < MAX_NUM_TOKENS_PADDED)
        return

    expert_offs = tl.arange(0, PADDED_NUM_EXPERTS)
    token_offs = tl.arange(0, HIST_BLOCK)
    expert_mask = expert_offs < NUM_EXPERTS

    counts = tl.zeros((PADDED_NUM_EXPERTS,), dtype=tl.int32)

    for token_start in range(0, NUMEL, HIST_BLOCK):
        offs = token_start + token_offs
        mask = offs < NUMEL
        expert_vals = tl.load(topk_ids_ptr + offs, mask=mask, other=NUM_EXPERTS).to(tl.int32)
        valid = mask & (expert_vals < NUM_EXPERTS)
        counts += tl.histogram(expert_vals, PADDED_NUM_EXPERTS, mask=valid)

    padded_counts = tl.where(
        expert_mask,
        tl.cdiv(counts, BLOCK_SIZE_M) * BLOCK_SIZE_M,
        0,
    )
    padded_cumsum = tl.cumsum(padded_counts, axis=0)
    starts = padded_cumsum - padded_counts
    total_tokens_post_pad = tl.sum(padded_counts, axis=0)

    tl.store(cumsum_ptr + expert_offs, starts, mask=expert_mask)
    tl.store(cumsum_ptr + NUM_EXPERTS, total_tokens_post_pad)
    tl.store(num_tokens_post_pad_ptr, total_tokens_post_pad)

    block_offs = tl.arange(0, HIST_BLOCK)
    for block_start in range(0, MAX_NUM_M_BLOCKS, HIST_BLOCK):
        block_ids = block_start + block_offs
        block_mask = block_ids < MAX_NUM_M_BLOCKS
        block_offsets = block_ids * BLOCK_SIZE_M
        block_expert = tl.full((HIST_BLOCK,), -1, dtype=tl.int32)

        for expert_idx in range(0, NUM_EXPERTS):
            start = tl.load(cumsum_ptr + expert_idx)
            end = tl.load(cumsum_ptr + expert_idx + 1)
            in_range = block_mask & (block_offsets >= start) & (block_offsets < end)
            block_expert = tl.where(in_range, expert_idx, block_expert)

        tl.store(expert_ids_ptr + block_ids, block_expert, mask=block_mask)


@triton.jit
def count_and_sort_expert_tokens_kernel(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    cumsum_ptr,
    BLOCK_SIZE: tl.constexpr,
    NUMEL: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    out0_ptr,
    out1_ptr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    token_offs = tl.arange(0, BLOCK_SIZE)

    token_start = pid * BLOCK_SIZE
    while token_start < NUMEL:
        offs = token_start + token_offs
        mask = offs < NUMEL
        expert_vals = tl.load(topk_ids_ptr + offs, mask=mask, other=NUM_EXPERTS).to(tl.int32)
        valid = mask & (expert_vals < NUM_EXPERTS)
        rank_post_pad = tl.atomic_add(cumsum_ptr + expert_vals, 1, mask=valid, sem="relaxed")
        tl.store(sorted_token_ids_ptr + rank_post_pad, offs, mask=valid)
        token_start += num_programs * BLOCK_SIZE
