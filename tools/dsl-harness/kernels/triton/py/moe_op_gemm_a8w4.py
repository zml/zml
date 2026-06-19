# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/matmul_ogs_details/_matmul_ogs.py

import torch
import triton
import triton.language as tl

@triton.jit
def _mxfp8_quant_op(x_grouped, QUANT_AXIS: tl.constexpr):
    """Shared MXFP8 (1x32 e8m0) scale derivation.

    Given a fp32 tile where the QUANT_AXIS dim is sized QUANT_BLOCK_SIZE (=32),
    returns (scale_e8m0, quant_scale): the per-group uint8 e8m0 scale and the
    matching fp32 multiplicative scale. Both outputs keep QUANT_AXIS with size 1
    so they broadcast against the input for in-place quantization.
    """
    amax = tl.max(tl.abs(x_grouped), axis=QUANT_AXIS, keep_dims=True)
    amax_i32 = amax.to(tl.int32, bitcast=True)
    amax_i32 = (amax_i32 + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax_p2 = amax_i32.to(tl.float32, bitcast=True)
    scale_unbiased = tl.log2(amax_p2).floor() - 8
    scale_unbiased = tl.clamp(scale_unbiased, min=-127, max=127)
    scale_e8m0 = (scale_unbiased.to(tl.int32) + 127).to(tl.uint8)
    quant_scale = tl.exp2(-scale_unbiased)
    return scale_e8m0, quant_scale

@triton.jit
def _swiglu(input, alpha, limit, ADD_RESIDUAL: tl.constexpr):
    """
    SwiGLU activation

    s = silu(gelu), then returns s * (linear + 1) if ADD_RESIDUAL else s * linear.
    if alpha=1.0, then this is the same as the SiLU activation.
    """
    gelu, linear = tl.split(tl.reshape(input, (input.shape[0], input.shape[1] // 2, 2)))
    gelu = gelu.to(tl.float32)
    if limit is not None:
        gelu = clip(gelu, limit, clip_lower=False)
    linear = linear.to(tl.float32)
    if limit is not None:
        linear = clip(linear, limit, clip_lower=True)
    s = gelu / (1 + tl.exp2(-1.44269504089 * alpha * gelu))
    if ADD_RESIDUAL:
        return tl.fma(s, linear, s)  # s * (linear + 1)
    else:
        return s * linear

@triton.jit
def _compute_static_fp8_quant(tensor, scale):
    tensor = tensor.to(tl.float32)
    tensor = tensor / scale
    tensor = tensor.to(tl.float8e4nv)
    return tensor

@triton.jit
def pid_grid(pid: int, num_pid_m: int, num_pid_n: int, GROUP_SIZE_M: tl.constexpr = 1):
    """
    Maps 1D pid to 2D grid coords (pid_m, pid_n).

    Args:
        - pid: 1D pid
        - num_pid_m: grid m size
        - num_pid_n: grid n size
        - GROUP_SIZE_M: tl.constexpr: default is 1
    """
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        tl.assume(group_size_m >= 0)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    return pid_m, pid_n


def matmul_launch_metadata(grid, kernel, args):
    ret = dict()
    M, N, K = None, args["N"], args["K"]
    Y, X, W = args["Y"], args["X"], args["W"]
    hist = args["ExptHist"]
    if hist is not None:
        n_rows = int(hist.float().mean())
        n_tokens = float(hist.sum())
        n_w_bytes = (W.numel() * W.element_size() // hist.numel()) * (hist > 0).sum()
    else:
        n_tokens = None
        n_w_bytes = W.numel() * W.element_size()

    def repr(s, x):
        return f"{s}={x}" if x is not None else f"E_{len(hist)}({s})={n_rows}"

    nbits = X.dtype.itemsize * 8
    ret["name"] = f"{kernel.name} [{repr('M', M)}, {repr('N', N)}, {repr('K', K)}]"
    gindx = args.get("GatherIndx", None)
    if gindx is not None:
        gindx = gindx.to(torch.int32)
        ret["name"] += "_layer1"
    else:
        ret["name"] += "_layer2"
    if args["B"] is not None:
        ret["name"] += "_bias"
    if args["APPLY_SWIGLU"]:
        ret["name"] += "_swiglu"
    if args["Quant_static_scale"] is not None:
        ret["name"] += "_quant"

    fM = n_tokens
    fK = K if K is not None else n_tokens
    ret[f"flops{nbits}"] = 2.0 * fM * N * fK

    n_x_bytes = X.numel() * X.element_size()
    n_y_bytes = Y.numel() * Y.element_size()
    if hist is not None:
        assert n_tokens is not None
        n_expts_act = args["N_EXPTS_ACT"]

        if gindx is not None:
            # recreate inverse GatherIndx.
            dst = torch.full_like(gindx, -1)
            idx = torch.arange(len(gindx), device=gindx.device, dtype=torch.int32)
            mask = gindx != -1
            dst[gindx[mask]] = idx[mask]
            n_read_rows = (dst.view((-1, n_expts_act)) != -1).any(dim=1).sum()
        else:
            n_read_rows = n_tokens
        n_x_bytes = n_read_rows * X.shape[-1] * X.element_size()
        n_y_bytes = n_tokens * Y.shape[-1] * Y.element_size()
    ret["bytes"] = int(n_x_bytes + n_y_bytes + n_w_bytes)

    return ret


# TODO: using aiter swizzle instead can lead to perf degradation in rare cases
@triton.jit
def xcd_swizzle(pid, domain_size, XCD_SWIZZLE: tl.constexpr):
    """
    Swizzle the program id based on integer XCD_SWIZZLE.
    This is useful for reording how blocks are ordered. A scheduler may, for example,
    assign sequential blocks 0, 1, 2, 3, ..., 8, 9, 10.. to its 8 hardware units 0, 1, 2, 3, ..., 0, 1, 2.
    This pattern may not be ideal for memory access, and it may be better to swizzle so the assignment
    becomes 0, 0, 0, 0, ..., 1, 1, 1, ... In the swizzled arrangement, sequential blocks are assigned to
    the same hardware unit.
    """
    # Number of pids per group in the new arrangement
    pids_per_group = domain_size // XCD_SWIZZLE
    extra_pid_groups = domain_size % XCD_SWIZZLE

    # Compute current current and local pid within the group
    group = pid % XCD_SWIZZLE
    local_pid = pid // XCD_SWIZZLE

    # Calculate new pid based on the new grouping
    new_pid = group * pids_per_group + min(group, extra_pid_groups) + local_pid
    return new_pid


@triton.jit
def unswizzle_mx_scale_cdna4(
    x,
    BLOCK_N: tl.constexpr,
    MX_SCALE_BLOCK_K: tl.constexpr,
    N_PRESHUFFLE_FACTOR: tl.constexpr = 32,
):
    x = x.reshape(BLOCK_N // N_PRESHUFFLE_FACTOR, MX_SCALE_BLOCK_K // 8, 4, 16, 2, 2, 1)
    x = x.permute(0, 5, 3, 1, 4, 2, 6)
    x = x.reshape(BLOCK_N, MX_SCALE_BLOCK_K)
    return x


@triton.jit(launch_metadata=matmul_launch_metadata)
def _moe_gemm_a8w4(
    Y,
    stride_y_k,
    stride_y_m,
    stride_y_n,
    X,
    stride_x_m,
    stride_x_k,
    XMxScale,
    stride_x_mx_m,
    stride_x_mx_k,
    W,
    stride_w_e,
    stride_w_k,
    stride_w_n,
    WMxScale,
    stride_w_mx_e,
    stride_w_mx_k,
    stride_w_mx_n,
    X_static_scale,
    Quant_static_scale,
    B,
    stride_b_e,  # Bias
    Gammas,
    N,
    K,  # shapes
    # expt data
    GatherIndx,
    ExptHist,
    ExptOffs,
    ExptOffsSum,
    ExptData,
    # true grid size
    grid_m,
    grid_n,
    # fused activation function
    APPLY_SWIGLU: tl.constexpr,
    alpha,
    limit,
    ACTIVATION_REDUCTION_N: tl.constexpr,
    SWIGLU_ADD_RESIDUAL: tl.constexpr,
    # MoE config
    N_EXPTS_ACT: tl.constexpr,
    # optimization config
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    XCD_SWIZZLE: tl.constexpr,
    # One of ["CDNA4", None]
    SWIZZLE_MX_SCALE: tl.constexpr,
    EVEN_K: tl.constexpr,
    MASK_K_LIMIT: tl.constexpr,
    SPLIT_K: tl.constexpr,
    W_CACHE_MODIFIER: tl.constexpr,
    UPCAST_INDICES: tl.constexpr = False,
    # Idea 1: fold per-1×32 MXFP8 (ue8m0 scale) group quant into write-back.
    # When HAS_MX_OUT=True, Y is fp8 e4m3 and YMxScale receives uint8 ue8m0
    # exponents at [m, n // 32]. Eliminates the standalone `downcast_to_mxfp`
    # launch between GEMM1 and GEMM2. Requires SPLIT_K==1 and OUT_BLOCK_N % 32 == 0.
    YMxScale=None,
    stride_y_mx_m=0,
    stride_y_mx_n=0,
    HAS_MX_OUT: tl.constexpr = False,
):
    tl.assume(stride_y_k >= 0)
    tl.assume(stride_y_m >= 0)
    tl.assume(stride_y_n >= 0)
    tl.assume(stride_x_m >= 0)
    tl.assume(stride_x_k >= 0)
    tl.assume(stride_w_e >= 0)
    tl.assume(stride_w_k >= 0)
    tl.assume(stride_w_n >= 0)
    if stride_x_mx_m is not None:
        tl.assume(stride_x_mx_m >= 0)
    if stride_x_mx_k is not None:
        tl.assume(stride_x_mx_k >= 0)
    if stride_w_mx_e is not None:
        tl.assume(stride_w_mx_e >= 0)
    if stride_w_mx_k is not None:
        tl.assume(stride_w_mx_k >= 0)
    if stride_w_mx_n is not None:
        tl.assume(stride_w_mx_n >= 0)
    if B is not None:
        tl.assume(stride_b_e >= 0)
    tl.assume(grid_m >= 0)
    tl.assume(grid_n >= 0)

    is_x_microscaled: tl.constexpr = XMxScale is not None
    MX_PACK_DIVISOR: tl.constexpr = 32
    w_type: tl.constexpr = W.dtype.element_ty
    tl.static_assert(w_type == tl.uint8, "mx_weight_ptr must be uint8 or fp8")
    tl.static_assert(
        WMxScale.dtype.element_ty == tl.uint8, "mx_scale_ptr must be uint8"
    )
    tl.static_assert(
        BLOCK_K % MX_PACK_DIVISOR == 0, "BLOCK_K must be a multiple of MX_PACK_DIVISOR"
    )
    x_type: tl.constexpr = X.dtype.element_ty
    if is_x_microscaled:
        tl.static_assert(x_type == tl.float8e4nv, "mx_act_ptr must be float8e4nv")
        tl.static_assert(
            XMxScale.dtype.element_ty == tl.uint8, "mx_scale_ptr must be uint8"
        )

    OUT_BLOCK_N: tl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    yN = N // ACTIVATION_REDUCTION_N

    pid = tl.program_id(0)
    if ExptOffsSum is not None and XCD_SWIZZLE > 1:
        # Determine how much padding there is on the expert data. This allows us to
        # know the true grid size and avoid processing padding tiles.
        padding_m = grid_m - tl.load(ExptOffsSum)
    else:
        padding_m: tl.constexpr = 0

    index_type: tl.constexpr = tl.int64 if UPCAST_INDICES else tl.int32

    unpadded_m = grid_m - padding_m
    tl.assume(unpadded_m >= 0)
    total_actual_tiles = unpadded_m * grid_n * SPLIT_K
    if padding_m > 0 and pid >= total_actual_tiles:
        return

    # swizzle program ids
    pid_emnk = pid
    if XCD_SWIZZLE != 1:
        pid_emnk = xcd_swizzle(pid_emnk, total_actual_tiles, XCD_SWIZZLE)
    # pid_e = pid_emnk // (unpadded_m * grid_n * SPLIT_K)
    pid_mnk = pid_emnk % (unpadded_m * grid_n * SPLIT_K)
    pid_k = pid_mnk % SPLIT_K
    pid_mn = pid_mnk // SPLIT_K
    pid_m, pid_n = pid_grid(pid_mn, unpadded_m, grid_n, GROUP_M)
    # For split-k, advance to the output k slice
    if SPLIT_K > 1:
        Y += pid_k.to(index_type) * stride_y_k
    # unpack expert data
    expt_data = tl.load(ExptData + pid_m)
    if expt_data == -1:
        return
    expt_id = expt_data & 0x0000FFFF
    block_id = expt_data >> 16
    M = tl.load(ExptHist + expt_id)
    start_m = tl.load(ExptOffs + expt_id)
    expt_id, block_id = expt_id.to(index_type), block_id.to(index_type)
    start_m = start_m.to(index_type)
    pid_n, pid_k = pid_n.to(index_type), pid_k.to(index_type)

    # A pointers
    offs_x_m = BLOCK_M * block_id + tl.arange(0, BLOCK_M)
    offs_x_m = tl.max_contiguous(tl.multiple_of(offs_x_m % M, BLOCK_M), BLOCK_M)
    if GatherIndx is None:
        X += start_m * stride_x_m
    else:
        GatherIndx += start_m
        # no needs to bounds-check here because `offs_x_m` wraps around M dim
        offs_x_m = tl.load(GatherIndx + offs_x_m) // N_EXPTS_ACT
    offs_x_k = BLOCK_K * pid_k + tl.arange(0, BLOCK_K)
    XPtrs = (
        X
        + offs_x_m.to(index_type)[:, None] * stride_x_m
        + offs_x_k.to(index_type)[None, :] * stride_x_k
    )

    W_K_DIVISOR: tl.constexpr = 2
    W_N_DIVISOR: tl.constexpr = 1
    PACKED_BLOCK_K_W: tl.constexpr = BLOCK_K // W_K_DIVISOR
    PACKED_BLOCK_N_W: tl.constexpr = BLOCK_N // W_N_DIVISOR
    MX_SCALE_BLOCK_K: tl.constexpr = BLOCK_K // MX_PACK_DIVISOR

    WMxScale += expt_id * stride_w_mx_e
    if SWIZZLE_MX_SCALE == "CDNA4_SCALE":
        tl.static_assert(stride_w_mx_k is not None)
        tl.static_assert(stride_w_mx_n is not None)
        NON_K_PRESHUFFLE_BLOCK_SIZE: tl.constexpr = 32
        PACKED_MX_BLOCK: tl.constexpr = MX_SCALE_BLOCK_K * NON_K_PRESHUFFLE_BLOCK_SIZE
        SCALE_BLOCK_N: tl.constexpr = BLOCK_N // NON_K_PRESHUFFLE_BLOCK_SIZE
    else:
        PACKED_MX_BLOCK: tl.constexpr = MX_SCALE_BLOCK_K
        SCALE_BLOCK_N: tl.constexpr = BLOCK_N
    offs_w_n_scale = (pid_n * SCALE_BLOCK_N + tl.arange(0, SCALE_BLOCK_N)) % N
    offs_w_n_scale = tl.max_contiguous(
        tl.multiple_of(offs_w_n_scale, SCALE_BLOCK_N), SCALE_BLOCK_N
    )
    # K dimension must be the last dimension for the scales
    offs_w_k_scale = PACKED_MX_BLOCK * pid_k + tl.arange(0, PACKED_MX_BLOCK)
    WMxScalePtrs = (
        WMxScale
        + offs_w_k_scale.to(index_type)[None, :] * stride_w_mx_k
        + offs_w_n_scale.to(index_type)[:, None] * stride_w_mx_n
    )

    # B pointers
    offs_w_n = pid_n * PACKED_BLOCK_N_W + tl.arange(0, PACKED_BLOCK_N_W)
    offs_w_n = tl.max_contiguous(
        tl.multiple_of(offs_w_n % (N // W_N_DIVISOR), PACKED_BLOCK_N_W),
        PACKED_BLOCK_N_W,
    )
    offs_w_k = PACKED_BLOCK_K_W * pid_k + tl.arange(0, PACKED_BLOCK_K_W)
    W += expt_id * stride_w_e
    WPtrs = W + (
        offs_w_k.to(index_type)[:, None] * stride_w_k
        + offs_w_n.to(index_type)[None, :] * stride_w_n
    )

    if is_x_microscaled:
        if GatherIndx is None:
            XMxScale += start_m * stride_x_mx_m
        offs_x_k_scale = MX_SCALE_BLOCK_K * pid_k + tl.arange(0, MX_SCALE_BLOCK_K)
        XMxScalePtrs = (
            XMxScale
            + offs_x_m.to(index_type)[:, None] * stride_x_mx_m
            + offs_x_k_scale.to(index_type)[None, :] * stride_x_mx_k
        )

    num_k_iter = tl.cdiv(K, BLOCK_K * SPLIT_K)
    if not EVEN_K:
        num_k_iter -= 1

    # compute output
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(num_k_iter):
        x = tl.load(XPtrs)
        w = tl.load(WPtrs, cache_modifier=W_CACHE_MODIFIER)

        if is_x_microscaled:
            x_scales = tl.load(XMxScalePtrs)
        else:
            x_scales = tl.full((BLOCK_M, MX_SCALE_BLOCK_K), 127, dtype=tl.uint8)
        if SWIZZLE_MX_SCALE == "CDNA4_SCALE":
            w_scales = unswizzle_mx_scale_cdna4(
                tl.load(WMxScalePtrs, cache_modifier=W_CACHE_MODIFIER),
                BLOCK_N,
                MX_SCALE_BLOCK_K,
            )
        else:
            w_scales = tl.load(WMxScalePtrs)

        acc = tl.dot_scaled(
            x, x_scales, "e4m3", w, w_scales, "e2m1", acc=acc, fast_math=True
        )

        WMxScalePtrs += (PACKED_MX_BLOCK * SPLIT_K) * stride_w_mx_k
        if is_x_microscaled:
            XMxScalePtrs += (MX_SCALE_BLOCK_K * SPLIT_K) * stride_x_mx_k

        XPtrs += (BLOCK_K * SPLIT_K) * stride_x_k
        WPtrs += (PACKED_BLOCK_K_W * SPLIT_K) * stride_w_k

    if not EVEN_K:
        mask_x_k = offs_x_k < MASK_K_LIMIT
        mask_w_k = offs_w_k < (MASK_K_LIMIT // W_K_DIVISOR)
        if SWIZZLE_MX_SCALE is None:
            mask_w_k_scale = offs_w_k_scale * MX_PACK_DIVISOR < MASK_K_LIMIT
        if is_x_microscaled:
            mask_x_k_scale = offs_x_k_scale * MX_PACK_DIVISOR < MASK_K_LIMIT

        x = tl.load(XPtrs, mask=mask_x_k[None, :], other=0.0)
        w = tl.load(
            WPtrs, mask=mask_w_k[:, None], other=0, cache_modifier=W_CACHE_MODIFIER
        )

        if is_x_microscaled:
            x_scales = tl.load(XMxScalePtrs, mask=mask_x_k_scale[None, :])
        else:
            x_scales = tl.full((BLOCK_M, MX_SCALE_BLOCK_K), 127, dtype=tl.uint8)
        if SWIZZLE_MX_SCALE == "CDNA4_SCALE":
            w_scales = unswizzle_mx_scale_cdna4(
                tl.load(WMxScalePtrs, cache_modifier=W_CACHE_MODIFIER),
                BLOCK_N,
                MX_SCALE_BLOCK_K,
            )
        else:
            w_scales = tl.load(WMxScalePtrs, mask=mask_w_k_scale[None, :])

        acc = tl.dot_scaled(
            x, x_scales, "e4m3", w, w_scales, "e2m1", acc=acc, fast_math=True
        )

    # scalar fp8 scale
    if X_static_scale is not None:
        acc = acc * tl.load(X_static_scale)
    # bias
    offs_m = BLOCK_M * block_id + tl.arange(0, BLOCK_M)
    offs_y_n = BLOCK_N * pid_n + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_y_n < N
    if B is not None:
        BPtrs = B + expt_id * stride_b_e + offs_y_n
        if pid_k == 0:
            bias = tl.load(BPtrs, mask=mask_n, other=0, cache_modifier=W_CACHE_MODIFIER)
        else:
            bias = tl.full([BLOCK_N], 0, dtype=tl.float32)
        acc = acc + bias[None, :]
    if APPLY_SWIGLU and SPLIT_K == 1:
        out = _swiglu(acc, alpha, limit, ADD_RESIDUAL=SWIGLU_ADD_RESIDUAL)
        tl.static_assert(
            out.shape[1] == OUT_BLOCK_N,
            f"Activation fn out.shape[1] ({out.shape[1]}) doesn't match computed OUT_BLOCK_N ({OUT_BLOCK_N})",
        )
        offs_y_n = OUT_BLOCK_N * pid_n + tl.arange(0, OUT_BLOCK_N)
        mask_n = offs_y_n < yN
    else:
        tl.static_assert(
            ACTIVATION_REDUCTION_N == 1,
            "Activation reduction must be 1 if no activation fn is provided",
        )
        out = acc
    if Gammas is not None:
        gammas = tl.load(Gammas + start_m + offs_m, mask=mask_m, other=0.0)
        out *= gammas[:, None]
    # quant
    if Quant_static_scale is not None:
        out = _compute_static_fp8_quant(out, tl.load(Quant_static_scale))
    # write-back
    Y += start_m * stride_y_m
    offs_y_m = offs_m
    YPtrs = (
        Y
        + offs_y_m.to(index_type)[:, None] * stride_y_m
        + offs_y_n.to(index_type)[None, :] * stride_y_n
    )
    mask = mask_m[:, None] & mask_n[None, :]
    if HAS_MX_OUT and SPLIT_K == 1:
        # Per-1×32 MXFP8 group quant emit. Mirrors the ue8m0 path in
        # `_fused_clamp_silu_mul_kernel`. OUT_BLOCK_N is the post-swiglu output
        # block size (BLOCK_N if no swiglu, BLOCK_N//2 with apply_swiglu).
        tl.static_assert(
            OUT_BLOCK_N % 32 == 0,
            "HAS_MX_OUT requires OUT_BLOCK_N % 32 == 0",
        )
        NUM_QB: tl.constexpr = OUT_BLOCK_N // 32
        out_safe = tl.where(mask, out, 0.0)
        out_3d = tl.reshape(out_safe, [BLOCK_M, NUM_QB, 32])
        # Reuse the shared MXFP8 (1x32 e8m0) scale derivation. It returns the
        # per-group uint8 e8m0 scale and the matching fp32 multiplicative scale,
        # both keeping the quant axis with size 1 so they broadcast over out_3d.
        scale_e8m0, quant_scale = _mxfp8_quant_op(out_3d, QUANT_AXIS=2)
        quant_tensor = out_3d * quant_scale
        quant_2d = tl.reshape(quant_tensor, [BLOCK_M, OUT_BLOCK_N])
        tl.store(YPtrs, quant_2d.to(Y.dtype.element_ty), mask=mask)
        scale_exp_2d = tl.reshape(scale_e8m0, [BLOCK_M, NUM_QB])
        offs_s_n = NUM_QB * pid_n + tl.arange(0, NUM_QB)
        mask_s_n = offs_s_n < tl.cdiv(yN, 32)
        YMxScalePtrs = (
            YMxScale
            + (start_m + offs_y_m).to(index_type)[:, None] * stride_y_mx_m
            + offs_s_n.to(index_type)[None, :] * stride_y_mx_n
        )
        tl.store(
            YMxScalePtrs,
            scale_exp_2d,
            mask=mask_m[:, None] & mask_s_n[None, :],
        )
    else:
        tl.store(YPtrs, out, mask=mask)
