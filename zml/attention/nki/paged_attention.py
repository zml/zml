import nki.isa as nisa
import nki.language as nl
import nki.typing as nt

NEG_INF = -3.3895313892515355e38


def paged_attention_2d(
    q,
    k_cache,
    v_cache,
    new_k,
    new_v,
    slot_mapping,
    block_table,
    seq_lens,
    query_start_len,
):
    """2D paged attention over split K/V pages using llmd/Triton metadata.

    Shapes:
      q:               [num_tokens, num_kv_heads, heads_per_kv, head_dim]
      k_cache/v_cache: [num_pages, page_size, num_kv_heads, head_dim]
      new_k/new_v:     [num_tokens, num_kv_heads, head_dim]
      slot_mapping:    [num_tokens] flattened cache slots, used by decode
      block_table:     [batch_size, max_num_pages]
      seq_lens:        [batch_size, 1]
      query_start_len: [batch_size + 1, 1]

    This NKI version supports batched decode (one query per batch row) and
    fixed-block batched prefill where each batch row owns the same number of
    compiled BLOCK_Q query blocks. It uses query_start_len/seq_lens-compatible
    metadata and expects the custom call to receive the local manual-computation
    shard.
    """
    num_tokens = q.shape[0]
    num_kv_heads = q.shape[1]
    heads_per_kv = q.shape[2]
    d_head = q.shape[3]
    page_size = k_cache.shape[1]
    max_num_pages = block_table.shape[1]
    batch_size = block_table.shape[0]
    scale = 1.0 / (d_head**0.5)

    assert q.shape[0] >= batch_size
    assert k_cache.shape == v_cache.shape
    assert new_k.shape == new_v.shape
    assert new_k.shape[0] == num_tokens
    assert new_k.shape[1] == num_kv_heads
    assert new_k.shape[2] == d_head
    assert k_cache.shape[2] == num_kv_heads
    assert k_cache.shape[3] == d_head
    assert seq_lens.shape[1] == 1
    assert query_start_len.shape[0] == batch_size + 1
    assert query_start_len.shape[1] == 1
    assert page_size <= 128, "page_size must fit the NKI partition dimension"

    out = nl.ndarray(q.shape, dtype=q.dtype, buffer=nl.shared_hbm)

    block_table_sbuf = nl.load(block_table.reshape((1, batch_size * max_num_pages)))
    seq_lens_sbuf = nl.load(seq_lens.reshape((1, batch_size)))

    if num_tokens == batch_size:
        k_cache_flat = k_cache.reshape(
            (k_cache.shape[0] * page_size, num_kv_heads, d_head)
        )
        v_cache_flat = v_cache.reshape(
            (v_cache.shape[0] * page_size, num_kv_heads, d_head)
        )
        slot_mapping_2d = slot_mapping.reshape((1, num_tokens))

        for token_idx in nl.sequential_range(batch_size):
            slot_sbuf = nl.ndarray((1, 1), dtype=slot_mapping.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=slot_sbuf,
                src=slot_mapping_2d[0, nl.ds(token_idx, 1)],
            )

            for kv_head in nl.affine_range(num_kv_heads):
                k_sbuf = nl.ndarray((1, d_head), dtype=new_k.dtype, buffer=nl.sbuf)
                v_sbuf = nl.ndarray((1, d_head), dtype=new_v.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=k_sbuf,
                    src=new_k[nl.ds(token_idx, 1), kv_head, :],
                )
                nisa.dma_copy(
                    dst=v_sbuf,
                    src=new_v[nl.ds(token_idx, 1), kv_head, :],
                )
                nisa.dma_copy(
                    dst=k_cache_flat.ap(
                        pattern=[[num_kv_heads * d_head, 1], [1, d_head]],
                        offset=kv_head * d_head,
                        scalar_offset=slot_sbuf,
                        indirect_dim=0,
                    ),
                    src=k_sbuf,
                    dge_mode=nisa.dge_mode.swdge,
                    oob_mode=nisa.oob_mode.skip,
                )
                nisa.dma_copy(
                    dst=v_cache_flat.ap(
                        pattern=[[num_kv_heads * d_head, 1], [1, d_head]],
                        offset=kv_head * d_head,
                        scalar_offset=slot_sbuf,
                        indirect_dim=0,
                    ),
                    src=v_sbuf,
                    dge_mode=nisa.dge_mode.swdge,
                    oob_mode=nisa.oob_mode.skip,
                )

        assert max_num_pages <= 128, (
            "decode max_num_pages must fit the NKI partition dimension"
        )

        DECODE_TOKEN_TILE = 128
        assert DECODE_TOKEN_TILE % page_size == 0, (
            "decode page_size must divide DECODE_TOKEN_TILE"
        )
        pages_per_tile = DECODE_TOKEN_TILE // page_size
        assert max_num_pages % pages_per_tile == 0, (
            "decode max_num_pages must be a multiple of pages_per_tile"
        )
        tile_tokens = DECODE_TOKEN_TILE
        num_page_tiles = max_num_pages // pages_per_tile

        page_positions = nl.ndarray((1, tile_tokens), dtype=nl.float32, buffer=nl.sbuf)
        seq_len_f32 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        qk = nl.ndarray((heads_per_kv, tile_tokens), dtype=nl.float32, buffer=nl.psum)
        qk_rounded = nl.ndarray(
            (heads_per_kv, tile_tokens), dtype=q.dtype, buffer=nl.sbuf
        )
        masked_qk = nl.ndarray(
            (heads_per_kv, tile_tokens), dtype=nl.float32, buffer=nl.sbuf
        )
        row_max = nl.ndarray((heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf)
        norm_tile = nl.ndarray(
            (heads_per_kv, tile_tokens), dtype=nl.float32, buffer=nl.sbuf
        )
        exp_tile = nl.ndarray(
            (heads_per_kv, tile_tokens), dtype=nl.float32, buffer=nl.sbuf
        )
        sum_row = nl.ndarray((heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf)
        inverse_sum_row = nl.ndarray(
            (heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        attn_out_psum = nl.ndarray(
            (heads_per_kv, d_head), dtype=nl.float32, buffer=nl.psum
        )
        attn_out_sbuf = nl.ndarray(
            (heads_per_kv, d_head), dtype=q.dtype, buffer=nl.sbuf
        )
        running_out = nl.ndarray(
            (heads_per_kv, d_head), dtype=nl.float32, buffer=nl.sbuf
        )
        old_max_delta = nl.ndarray(
            (heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        old_max_scale = nl.ndarray(
            (heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        scaled_sum_row = nl.ndarray(
            (heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        scaled_running_out = nl.ndarray(
            (heads_per_kv, d_head), dtype=nl.float32, buffer=nl.sbuf
        )
        tile_max = nl.ndarray((heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf)
        tile_sum = nl.ndarray((heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf)
        new_row_max = nl.ndarray((heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf)
        new_sum_row = nl.ndarray((heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf)
        pv_sbuf = nl.ndarray(
            (heads_per_kv, d_head), dtype=nl.float32, buffer=nl.sbuf
        )
        neg_tile = nl.full(
            (heads_per_kv, tile_tokens),
            NEG_INF,
            dtype=nl.float32,
            buffer=nl.sbuf,
        )

        for q_idx in nl.sequential_range(num_tokens):
            seq_idx = q_idx
            seq_len = seq_lens_sbuf[0, seq_idx]

            for kv_head in nl.sequential_range(num_kv_heads):
                q_sbuf = nl.load_transpose2d(
                    q[q_idx, kv_head, :, :],
                    dtype=q.dtype,
                )
                q_scaled = nl.ndarray(
                    (d_head, heads_per_kv), dtype=q.dtype, buffer=nl.sbuf
                )
                nisa.tensor_scalar(
                    dst=q_scaled,
                    data=q_sbuf,
                    op0=nl.multiply,
                    operand0=scale,
                )

                nisa.memset(row_max, NEG_INF)
                nisa.memset(sum_row, 0.0)
                nisa.memset(running_out, 0.0)
                for page_tile_idx in nl.sequential_range(num_page_tiles):
                    page_start = page_tile_idx * tile_tokens

                    nisa.iota(page_positions, [[1, tile_tokens]], offset=page_start)
                    nisa.tensor_copy(dst=seq_len_f32, src=seq_len)
                    k_sbuf = nl.ndarray(
                        (d_head, tile_tokens), dtype=k_cache.dtype, buffer=nl.sbuf
                    )
                    nisa.memset(k_sbuf, 0.0)
                    for page_in_tile in nl.affine_range(pages_per_tile):
                        logical_page_idx = page_tile_idx * pages_per_tile + page_in_tile
                        tile_offset = page_in_tile * page_size
                        physical_page = block_table_sbuf[
                            0, nl.ds(seq_idx * max_num_pages + logical_page_idx, 1)
                        ]
                        nisa.dma_copy(
                            k_sbuf[0:d_head, nl.ds(tile_offset, page_size)],
                            k_cache.ap(
                                pattern=[[1, d_head], [num_kv_heads * d_head, page_size]],
                                offset=kv_head * d_head,
                                scalar_offset=physical_page,
                                indirect_dim=0,
                            ),
                            dge_mode=nisa.dge_mode.swdge,
                            oob_mode=nisa.oob_mode.skip,
                        )
                    nisa.memset(qk, 0.0)
                    nisa.nc_matmul(dst=qk, stationary=q_scaled, moving=k_sbuf)
                    nisa.tensor_copy(dst=qk_rounded, src=qk)

                    valid_seq = nl.less(
                        page_positions,
                        nl.broadcast_to(seq_len_f32, page_positions.shape),
                    )
                    masked_qk = nl.where(
                        nl.broadcast_to(valid_seq, masked_qk.shape),
                        qk_rounded,
                        neg_tile,
                        dtype=nl.float32,
                    )
                    nisa.tensor_reduce(
                        dst=tile_max,
                        op=nl.maximum,
                        data=masked_qk,
                        axis=(1,),
                    )
                    nisa.tensor_tensor(
                        dst=new_row_max,
                        data1=row_max,
                        data2=tile_max,
                        op=nl.maximum,
                    )
                    nisa.tensor_scalar(
                        dst=old_max_delta,
                        data=row_max,
                        op0=nl.subtract,
                        operand0=new_row_max,
                    )
                    nisa.activation(
                        dst=old_max_scale,
                        op=nl.exp,
                        data=old_max_delta,
                    )
                    nisa.tensor_scalar(
                        dst=norm_tile,
                        data=masked_qk,
                        op0=nl.subtract,
                        operand0=new_row_max,
                    )
                    nisa.activation(dst=exp_tile, op=nl.exp, data=norm_tile)
                    nisa.tensor_reduce(
                        dst=tile_sum,
                        op=nl.add,
                        data=exp_tile,
                        axis=(1,),
                    )
                    nisa.tensor_scalar(
                        dst=scaled_sum_row,
                        data=sum_row,
                        op0=nl.multiply,
                        operand0=old_max_scale,
                    )
                    nisa.tensor_tensor(
                        dst=new_sum_row,
                        data1=scaled_sum_row,
                        data2=tile_sum,
                        op=nl.add,
                    )
                    nisa.tensor_scalar(
                        dst=scaled_running_out,
                        data=running_out,
                        op0=nl.multiply,
                        operand0=old_max_scale,
                    )

                    prob_t_psum = nl.ndarray(
                        (tile_tokens, heads_per_kv), dtype=nl.float32, buffer=nl.psum
                    )
                    prob_t = nl.ndarray(
                        (tile_tokens, heads_per_kv), dtype=v_cache.dtype, buffer=nl.sbuf
                    )
                    nisa.nc_transpose(dst=prob_t_psum, data=exp_tile)
                    nisa.tensor_copy(dst=prob_t, src=prob_t_psum)

                    v_sbuf = nl.ndarray(
                        (tile_tokens, d_head), dtype=v_cache.dtype, buffer=nl.sbuf
                    )
                    nisa.memset(v_sbuf, 0.0)
                    for page_in_tile in nl.affine_range(pages_per_tile):
                        logical_page_idx = page_tile_idx * pages_per_tile + page_in_tile
                        tile_offset = page_in_tile * page_size
                        physical_page = block_table_sbuf[
                            0, nl.ds(seq_idx * max_num_pages + logical_page_idx, 1)
                        ]
                        nisa.dma_copy(
                            v_sbuf[nl.ds(tile_offset, page_size), 0:d_head],
                            v_cache.ap(
                                pattern=[[num_kv_heads * d_head, page_size], [1, d_head]],
                                offset=kv_head * d_head,
                                scalar_offset=physical_page,
                                indirect_dim=0,
                            ),
                            dge_mode=nisa.dge_mode.swdge,
                            oob_mode=nisa.oob_mode.skip,
                        )
                    nisa.memset(attn_out_psum, 0.0)
                    nisa.nc_matmul(
                        dst=attn_out_psum,
                        stationary=prob_t,
                        moving=v_sbuf,
                    )
                    nisa.tensor_copy(dst=pv_sbuf, src=attn_out_psum)
                    nisa.tensor_tensor(
                        dst=running_out,
                        data1=scaled_running_out,
                        data2=pv_sbuf,
                        op=nl.add,
                    )
                    nisa.tensor_copy(dst=row_max, src=new_row_max)
                    nisa.tensor_copy(dst=sum_row, src=new_sum_row)

                nisa.reciprocal(dst=inverse_sum_row, data=sum_row)
                nisa.tensor_scalar(
                    dst=running_out,
                    data=running_out,
                    op0=nl.multiply,
                    operand0=inverse_sum_row,
                )
                nisa.tensor_copy(dst=attn_out_sbuf, src=running_out)
                nl.store(dst=out[q_idx, kv_head, :, :], value=attn_out_sbuf)
    else:
        BLOCK_Q = 128
        assert num_tokens % BLOCK_Q == 0, (
            f"prefill/mixed num_tokens ({num_tokens}) must be a multiple of BLOCK_Q ({BLOCK_Q})"
        )
        assert max_num_pages <= 128, (
            "max_num_pages must fit the NKI partition dimension"
        )

        num_q_blocks = num_tokens // BLOCK_Q
        query_start_len_sbuf = nl.load(query_start_len.reshape((1, batch_size + 1)))

        for kv_head in nl.affine_range(num_kv_heads):
            for head_group in nl.affine_range(heads_per_kv):
                for q_block_idx in nl.affine_range(num_q_blocks):
                    q_start = q_block_idx * BLOCK_Q
                    q_sbuf = nl.ndarray(
                        (d_head, BLOCK_Q), dtype=q.dtype, buffer=nl.sbuf
                    )
                    nisa.dma_transpose(
                        q_sbuf,
                        q[nl.ds(q_start, BLOCK_Q), kv_head, head_group, :],
                    )
                    q_scaled = nl.ndarray(
                        (d_head, BLOCK_Q), dtype=q.dtype, buffer=nl.sbuf
                    )
                    nisa.tensor_scalar(
                        dst=q_scaled,
                        data=q_sbuf,
                        op0=nl.multiply,
                        operand0=scale,
                    )

                    block_out = nl.ndarray(
                        (BLOCK_Q, d_head), dtype=nl.float32, buffer=nl.sbuf
                    )
                    block_out_next = nl.ndarray(
                        (BLOCK_Q, d_head), dtype=nl.float32, buffer=nl.sbuf
                    )
                    neg_tile = nl.full(
                        (BLOCK_Q, page_size),
                        NEG_INF,
                        dtype=nl.float32,
                        buffer=nl.sbuf,
                    )
                    zero_row = nl.full(
                        (BLOCK_Q, 1), 0.0, dtype=nl.float32, buffer=nl.sbuf
                    )
                    one_row = nl.full(
                        (BLOCK_Q, 1), 1.0, dtype=nl.float32, buffer=nl.sbuf
                    )
                    zero_out = nl.full(
                        (BLOCK_Q, d_head), 0.0, dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.memset(block_out, 0.0)

                    for seq_idx in nl.sequential_range(batch_size):
                        q_start_seq = query_start_len_sbuf[0, nl.ds(seq_idx, 1)]
                        q_end_seq = query_start_len_sbuf[0, nl.ds(seq_idx + 1, 1)]
                        seq_len = seq_lens_sbuf[0, nl.ds(seq_idx, 1)]

                        qk_psum = nl.ndarray(
                            (BLOCK_Q, page_size), dtype=nl.float32, buffer=nl.psum
                        )
                        qk_rounded = nl.ndarray(
                            (BLOCK_Q, page_size), dtype=q.dtype, buffer=nl.sbuf
                        )
                        masked_qk = nl.ndarray(
                            (BLOCK_Q, page_size), dtype=nl.float32, buffer=nl.sbuf
                        )
                        row_max = nl.ndarray((BLOCK_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                        sum_row = nl.ndarray((BLOCK_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                        running_out = nl.ndarray(
                            (BLOCK_Q, d_head), dtype=nl.float32, buffer=nl.sbuf
                        )
                        old_max_delta = nl.ndarray(
                            (BLOCK_Q, 1), dtype=nl.float32, buffer=nl.sbuf
                        )
                        old_max_scale = nl.ndarray(
                            (BLOCK_Q, 1), dtype=nl.float32, buffer=nl.sbuf
                        )
                        scaled_sum_row = nl.ndarray(
                            (BLOCK_Q, 1), dtype=nl.float32, buffer=nl.sbuf
                        )
                        scaled_running_out = nl.ndarray(
                            (BLOCK_Q, d_head), dtype=nl.float32, buffer=nl.sbuf
                        )
                        tile_max = nl.ndarray((BLOCK_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                        tile_sum = nl.ndarray((BLOCK_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                        new_row_max = nl.ndarray(
                            (BLOCK_Q, 1), dtype=nl.float32, buffer=nl.sbuf
                        )
                        new_sum_row = nl.ndarray(
                            (BLOCK_Q, 1), dtype=nl.float32, buffer=nl.sbuf
                        )
                        norm_tile = nl.ndarray(
                            (BLOCK_Q, page_size), dtype=nl.float32, buffer=nl.sbuf
                        )
                        exp_tile = nl.ndarray(
                            (BLOCK_Q, page_size), dtype=nl.float32, buffer=nl.sbuf
                        )
                        q_positions_f32 = nl.ndarray(
                            (BLOCK_Q, page_size),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        q_positions_abs = nl.ndarray(
                            (BLOCK_Q, page_size),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        combined_mask = nl.ndarray(
                            (BLOCK_Q, page_size),
                            dtype=seq_lens.dtype,
                            buffer=nl.sbuf,
                        )
                        q_row_positions_f32 = nl.ndarray(
                            (BLOCK_Q, 1),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        key_positions_f32 = nl.ndarray(
                            (BLOCK_Q, page_size),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        context_tail_bcast = nl.ndarray(
                            (BLOCK_Q, page_size),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        attn_out_psum = nl.ndarray(
                            (BLOCK_Q, d_head), dtype=nl.float32, buffer=nl.psum
                        )
                        pv_sbuf = nl.ndarray(
                            (BLOCK_Q, d_head), dtype=nl.float32, buffer=nl.sbuf
                        )
                        prob_t_psum = nl.ndarray(
                            (page_size, BLOCK_Q), dtype=nl.float32, buffer=nl.psum
                        )
                        prob_t = nl.ndarray(
                            (page_size, BLOCK_Q), dtype=v_cache.dtype, buffer=nl.sbuf
                        )
                        v_sbuf = nl.ndarray(
                            (page_size, d_head), dtype=v_cache.dtype, buffer=nl.sbuf
                        )
                        seq_len_f32 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
                        q_start_f32 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
                        q_end_f32 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
                        context_tail = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=seq_len_f32, src=seq_len)
                        nisa.tensor_copy(dst=q_start_f32, src=q_start_seq)
                        nisa.tensor_copy(dst=q_end_f32, src=q_end_seq)
                        nisa.iota(q_row_positions_f32, [[1, 1]], offset=q_start)
                        valid_q_row = nl.logical_and(
                            nl.greater_equal(
                                q_row_positions_f32,
                                nl.broadcast_to(q_start_f32, q_row_positions_f32.shape),
                            ),
                            nl.less(
                                q_row_positions_f32,
                                nl.broadcast_to(q_end_f32, q_row_positions_f32.shape),
                            ),
                        )
                        nisa.tensor_scalar(
                            dst=context_tail,
                            data=seq_len_f32,
                            op0=nl.subtract,
                            operand0=q_end_f32,
                        )
                        nisa.iota(
                            q_positions_f32,
                            [[0, page_size]],
                            offset=q_start,
                            channel_multiplier=1,
                        )
                        context_tail_bcast = nl.broadcast_to(
                            context_tail, context_tail_bcast.shape
                        )
                        nisa.tensor_tensor(
                            dst=q_positions_abs,
                            data1=q_positions_f32,
                            data2=context_tail_bcast,
                            op=nl.add,
                        )

                        nisa.memset(row_max, NEG_INF)
                        row_max = nl.where(
                            valid_q_row, row_max, zero_row, dtype=nl.float32
                        )
                        nisa.memset(sum_row, 0.0)
                        nisa.memset(running_out, 0.0)

                        for logical_page_idx in nl.affine_range(max_num_pages):
                            page_start = logical_page_idx * page_size
                            physical_page = block_table_sbuf[
                                0, nl.ds(seq_idx * max_num_pages + logical_page_idx, 1)
                            ]

                            k_sbuf = nl.ndarray(
                                (d_head, page_size), dtype=k_cache.dtype, buffer=nl.sbuf
                            )
                            nisa.memset(k_sbuf, 0.0)
                            nisa.dma_copy(
                                k_sbuf,
                                k_cache.ap(
                                    pattern=[
                                        [1, d_head],
                                        [num_kv_heads * d_head, page_size],
                                    ],
                                    offset=kv_head * d_head,
                                    scalar_offset=physical_page,
                                    indirect_dim=0,
                                ),
                                dge_mode=nisa.dge_mode.swdge,
                                oob_mode=nisa.oob_mode.skip,
                            )
                            nisa.memset(qk_psum, 0.0)
                            nisa.nc_matmul(
                                dst=qk_psum,
                                stationary=q_scaled,
                                moving=k_sbuf,
                            )
                            nisa.tensor_copy(dst=qk_rounded, src=qk_psum)

                            nisa.iota(
                                key_positions_f32,
                                [[1, page_size]],
                                offset=page_start,
                            )
                            causal_mask = nl.greater_equal(
                                q_positions_abs, key_positions_f32
                            )
                            combined_mask = nl.logical_and(
                                causal_mask,
                                nl.broadcast_to(valid_q_row, combined_mask.shape),
                            )
                            masked_qk = nl.where(
                                combined_mask, qk_rounded, neg_tile, dtype=nl.float32
                            )
                            nisa.tensor_reduce(
                                dst=tile_max,
                                op=nl.maximum,
                                data=masked_qk,
                                axis=(1,),
                            )
                            nisa.tensor_tensor(
                                dst=new_row_max,
                                data1=row_max,
                                data2=tile_max,
                                op=nl.maximum,
                            )
                            nisa.tensor_scalar(
                                dst=old_max_delta,
                                data=row_max,
                                op0=nl.subtract,
                                operand0=new_row_max,
                            )
                            nisa.activation(
                                dst=old_max_scale,
                                op=nl.exp,
                                data=old_max_delta,
                            )
                            nisa.tensor_scalar(
                                dst=norm_tile,
                                data=masked_qk,
                                op0=nl.subtract,
                                operand0=new_row_max,
                            )
                            nisa.activation(dst=exp_tile, op=nl.exp, data=norm_tile)
                            nisa.tensor_reduce(
                                dst=tile_sum,
                                op=nl.add,
                                data=exp_tile,
                                axis=(1,),
                            )
                            nisa.tensor_scalar(
                                dst=scaled_sum_row,
                                data=sum_row,
                                op0=nl.multiply,
                                operand0=old_max_scale,
                            )
                            nisa.tensor_tensor(
                                dst=new_sum_row,
                                data1=scaled_sum_row,
                                data2=tile_sum,
                                op=nl.add,
                            )
                            nisa.tensor_scalar(
                                dst=scaled_running_out,
                                data=running_out,
                                op0=nl.multiply,
                                operand0=old_max_scale,
                            )
                            nisa.nc_transpose(dst=prob_t_psum, data=exp_tile)
                            nisa.tensor_copy(dst=prob_t, src=prob_t_psum)

                            nisa.memset(v_sbuf, 0.0)
                            nisa.dma_copy(
                                v_sbuf,
                                v_cache.ap(
                                    pattern=[
                                        [num_kv_heads * d_head, page_size],
                                        [1, d_head],
                                    ],
                                    offset=kv_head * d_head,
                                    scalar_offset=physical_page,
                                    indirect_dim=0,
                                ),
                                dge_mode=nisa.dge_mode.swdge,
                                oob_mode=nisa.oob_mode.skip,
                            )
                            nisa.memset(attn_out_psum, 0.0)
                            nisa.nc_matmul(
                                dst=attn_out_psum,
                                stationary=prob_t,
                                moving=v_sbuf,
                            )
                            nisa.tensor_copy(dst=pv_sbuf, src=attn_out_psum)
                            nisa.tensor_tensor(
                                dst=running_out,
                                data1=scaled_running_out,
                                data2=pv_sbuf,
                                op=nl.add,
                            )
                            nisa.tensor_copy(dst=row_max, src=new_row_max)
                            nisa.tensor_copy(dst=sum_row, src=new_sum_row)

                        safe_sum_row = nl.where(
                            valid_q_row, sum_row, one_row, dtype=nl.float32
                        )
                        safe_running_out = nl.where(
                            nl.broadcast_to(valid_q_row, running_out.shape),
                            running_out,
                            zero_out,
                            dtype=nl.float32,
                        )
                        inverse_sum_row = nl.ndarray(
                            (BLOCK_Q, 1), dtype=nl.float32, buffer=nl.sbuf
                        )
                        nisa.reciprocal(dst=inverse_sum_row, data=safe_sum_row)
                        nisa.tensor_scalar(
                            dst=safe_running_out,
                            data=safe_running_out,
                            op0=nl.multiply,
                            operand0=inverse_sum_row,
                        )
                        block_out_next = nl.where(
                            nl.broadcast_to(valid_q_row, block_out.shape),
                            safe_running_out,
                            block_out,
                            dtype=nl.float32,
                        )
                        nisa.tensor_copy(dst=block_out, src=block_out_next)

                    attn_out_sbuf = nl.ndarray(
                        (BLOCK_Q, d_head), dtype=q.dtype, buffer=nl.sbuf
                    )
                    nisa.tensor_copy(dst=attn_out_sbuf, src=block_out)
                    nl.store(
                        dst=out[nl.ds(q_start, BLOCK_Q), kv_head, head_group, :],
                        value=attn_out_sbuf,
                    )

    return out


def paged_kv_cache_update(
    k_cache: nt.tensor,
    v_cache: nt.tensor,
    new_k,
    new_v,
    slot_mapping,
    query_start_len,
):
    """Update split paged K/V cache for live NeuronContext rows only.

    Shapes:
      k_cache/v_cache: [num_pages, page_size, num_kv_heads, head_dim]
      new_k/new_v:     [compiled_tokens, num_kv_heads, head_dim]
      slot_mapping:    [compiled_tokens]
      query_start_len: [batch_size + 1, 1]

    Padded slot_mapping rows may contain maxInt(i32). The update path keeps
    those rows inside the Neuron custom call and relies on AP destination
    writes with oob_mode=skip, matching scatter-drop semantics without lowering
    the update through HLO scatter/DGE.
    """
    num_pages = k_cache.shape[0]
    page_size = k_cache.shape[1]
    num_kv_heads = k_cache.shape[2]
    d_head = k_cache.shape[3]
    batch_size = query_start_len.shape[0] - 1

    assert k_cache.shape == v_cache.shape
    assert new_k.shape == new_v.shape
    assert new_k.shape[1] == num_kv_heads
    assert new_k.shape[2] == d_head
    assert slot_mapping.shape[0] == new_k.shape[0]
    assert query_start_len.shape[1] == 1

    _ = query_start_len
    _ = batch_size
    compiled_tokens = new_k.shape[0]
    k_cache_flat = k_cache.reshape((num_pages * page_size, num_kv_heads, d_head))
    v_cache_flat = v_cache.reshape((num_pages * page_size, num_kv_heads, d_head))
    slot_mapping_2d = slot_mapping.reshape((compiled_tokens, 1))

    for token_idx in nl.sequential_range(compiled_tokens):
        slot_sbuf = nl.ndarray((1, 1), dtype=slot_mapping.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=slot_sbuf,
            src=slot_mapping_2d[nl.ds(token_idx, 1), :],
        )
        for kv_head in nl.affine_range(num_kv_heads):
            k_sbuf = nl.ndarray((1, d_head), dtype=new_k.dtype, buffer=nl.sbuf)
            v_sbuf = nl.ndarray((1, d_head), dtype=new_v.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=k_sbuf,
                src=new_k[nl.ds(token_idx, 1), kv_head, :],
            )
            nisa.dma_copy(
                dst=v_sbuf,
                src=new_v[nl.ds(token_idx, 1), kv_head, :],
            )
            nisa.dma_copy(
                dst=k_cache_flat.ap(
                    pattern=[[num_kv_heads * d_head, 1], [1, d_head]],
                    offset=kv_head * d_head,
                    scalar_offset=slot_sbuf,
                    indirect_dim=0,
                ),
                src=k_sbuf,
                dge_mode=nisa.dge_mode.swdge,
                oob_mode=nisa.oob_mode.skip,
            )
            nisa.dma_copy(
                dst=v_cache_flat.ap(
                    pattern=[[num_kv_heads * d_head, 1], [1, d_head]],
                    offset=kv_head * d_head,
                    scalar_offset=slot_sbuf,
                    indirect_dim=0,
                ),
                src=v_sbuf,
                dge_mode=nisa.dge_mode.swdge,
                oob_mode=nisa.oob_mode.skip,
            )

    return k_cache, v_cache
