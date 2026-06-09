"""ZML paged-attention NKI kernels.

Entrypoints:

  paged_attention_decode_2d
    Writes the current decode token into the paged K/V cache, then attends over
    the updated cache. Each KV stream is shared across the GQA query heads that
    reference that KV head.

  paged_attention_2d
    Attends over compact prefill/mixed query rows. query_start_len maps each
    sequence to its live compact row span, so the kernel can skip empty spans
    and mask only the tail rows of the final query tile.

  paged_kv_cache_update
    Copies compact K/V update rows into flattened cache slots. V uses the normal
    [page, page_offset, hkv, head_dim] physical order. K is private to the NKI
    backend and uses [page, hkv, head_dim, page_offset] physical order so QK can
    load K pages directly in matmul-ready [head_dim, page_size] layout. Invalid
    slot_mapping rows are represented as out-of-range AP destinations and are
    ignored by oob_mode=skip.

Data ownership:
  * block_table maps logical pages to physical cache pages.
  * seq_lens defines which key positions are valid for each sequence.
  * query_start_len defines the compact query-row schedule for prefill/mixed.
  * slot_mapping defines cache write destinations, not attention order.
  * The online softmax state is updated during K/V tile streaming, so no HBM
    softmax scratch buffers are needed.
"""

import nki.isa as nisa
import nki.language as nl
import nki.typing as nt

NEG_INF = -3.3895313892515355e38


# Shared metadata and page-load helpers.


def page_shift(page_size):
    if page_size == 16:
        return 4
    if page_size == 32:
        return 5
    if page_size == 64:
        return 6
    assert False, "paged attention page_size must be one of 16, 32, or 64"


def private_k_slot_offset(slot_sbuf, num_pages, page_size, num_kv_heads, d_head):
    """Map normal flat slot to private K-cache flat element offset.

    slot_mapping uses the public flat slot convention:
      slot = physical_page * page_size + page_offset

    K cache storage is private to this NKI backend:
      [page, hkv, head_dim, page_offset]

    Invalid/padded slots are first mapped to zero before arithmetic to avoid
    int32 overflow, then replaced by one-past-end for oob_mode=skip.
    """
    valid_slot = nl.less(slot_sbuf, num_pages * page_size)
    zero_i32 = nl.full((1, 1), 0, dtype=slot_sbuf.dtype, buffer=nl.sbuf)
    invalid_private_offset = nl.full(
        (1, 1),
        num_pages * page_size * num_kv_heads * d_head,
        dtype=slot_sbuf.dtype,
        buffer=nl.sbuf,
    )
    safe_slot = nl.where(valid_slot, slot_sbuf, zero_i32, dtype=slot_sbuf.dtype)

    physical_page = nl.ndarray((1, 1), dtype=slot_sbuf.dtype, buffer=nl.sbuf)
    page_base = nl.ndarray((1, 1), dtype=slot_sbuf.dtype, buffer=nl.sbuf)
    page_offset = nl.ndarray((1, 1), dtype=slot_sbuf.dtype, buffer=nl.sbuf)
    private_offset = nl.ndarray((1, 1), dtype=slot_sbuf.dtype, buffer=nl.sbuf)

    nisa.tensor_scalar(
        dst=physical_page,
        data=safe_slot,
        op0=nl.right_shift,
        operand0=page_shift(page_size),
    )
    nisa.tensor_scalar(
        dst=page_base,
        data=physical_page,
        op0=nl.multiply,
        operand0=page_size,
    )
    nisa.tensor_tensor(
        dst=page_offset,
        data1=safe_slot,
        data2=page_base,
        op=nl.subtract,
    )
    nisa.tensor_scalar(
        dst=private_offset,
        data=physical_page,
        op0=nl.multiply,
        operand0=page_size * num_kv_heads * d_head,
    )
    nisa.tensor_tensor(
        dst=private_offset,
        data1=private_offset,
        data2=page_offset,
        op=nl.add,
    )
    return nl.where(
        valid_slot,
        private_offset,
        invalid_private_offset,
        dtype=slot_sbuf.dtype,
    )


def load_static_attention_metadata(
    block_table,
    seq_lens,
    query_start_len,
    batch_size,
    max_num_pages,
):
    """Load prefill/mixed metadata that is reused across Q/K/V loops."""
    block_table_sbuf = nl.ndarray(
        (1, batch_size * max_num_pages), dtype=block_table.dtype, buffer=nl.sbuf
    )
    seq_lens_sbuf = nl.ndarray((1, batch_size), dtype=seq_lens.dtype, buffer=nl.sbuf)
    query_start_len_sbuf = nl.ndarray(
        (1, batch_size + 1), dtype=query_start_len.dtype, buffer=nl.sbuf
    )
    nisa.dma_copy(
        dst=block_table_sbuf,
        src=block_table.reshape((1, batch_size * max_num_pages))[
            0:1, 0 : batch_size * max_num_pages
        ],
    )
    nisa.dma_copy(
        dst=seq_lens_sbuf,
        src=seq_lens.reshape((1, batch_size))[0:1, 0:batch_size],
    )
    nisa.dma_copy(
        dst=query_start_len_sbuf,
        src=query_start_len.reshape((1, batch_size + 1))[0:1, 0 : batch_size + 1],
    )
    return block_table_sbuf, seq_lens_sbuf, query_start_len_sbuf


def make_page_offsets(pages_per_k_tile, page_size):
    """Build logical token starts for each page inside a K tile."""
    page_offsets = nl.ndarray((1, pages_per_k_tile), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(
        page_offsets,
        [[1, pages_per_k_tile]],
        channel_multiplier=page_size,
    )
    return page_offsets


def choose_page_aligned_tile(max_num_pages, page_size, max_tile_tokens):
    """Pick a K tile that covers an integer number of page-table entries.

    The K stream advances by whole pages. Choosing the largest page-aligned tile
    allowed by the caller reduces online-softmax iterations and keeps a simple
    page loop for K and V loads.
    """
    pages_per_tile = min(max_num_pages, max_tile_tokens // page_size)
    for candidate in range(pages_per_tile, 0, -1):
        if max_num_pages % candidate == 0:
            pages_per_tile = candidate
            break
    return pages_per_tile * page_size


def choose_k_tiles_per_segment(num_k_tiles, max_tiles_per_segment):
    """Pick a segment size that divides the compiled K-tile stream."""
    tiles_per_segment = min(num_k_tiles, max_tiles_per_segment)
    for candidate in range(tiles_per_segment, 0, -1):
        if num_k_tiles % candidate == 0:
            return candidate
    return 1


def build_k_tile_pages(
    block_table_sbuf,
    seq_len_f32,
    seq_idx,
    k_tile_idx,
    page_offsets,
    inactive_pages,
    max_num_pages,
    pages_per_k_tile,
    k_tile_tokens,
):
    """Build active physical pages for one K tile.

    The returned table is reused for both K and V loads. Inactive logical pages
    are converted to -1 so AP loads can use oob_mode=skip without trusting that
    padded block-table entries name valid physical pages.
    """
    physical_pages = nl.ndarray(
        (1, pages_per_k_tile), dtype=block_table_sbuf.dtype, buffer=nl.sbuf
    )
    page_starts = nl.ndarray((1, pages_per_k_tile), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=physical_pages,
        src=block_table_sbuf[
            nl.ds(0, 1),
            nl.ds(
                seq_idx * max_num_pages + k_tile_idx * pages_per_k_tile,
                pages_per_k_tile,
            ),
        ],
    )
    nisa.tensor_scalar(
        dst=page_starts,
        data=page_offsets,
        op0=nl.add,
        operand0=k_tile_idx * k_tile_tokens,
    )
    page_live = nl.greater(nl.broadcast_to(seq_len_f32, page_starts.shape), page_starts)
    return nl.where(
        page_live, physical_pages, inactive_pages, dtype=block_table_sbuf.dtype
    )


def build_k_segment_pages(
    block_table_sbuf,
    seq_len_f32,
    seq_idx,
    segment_idx,
    page_offsets,
    inactive_pages,
    max_num_pages,
    pages_per_segment,
    segment_tokens,
):
    """Build active physical pages once for a larger decode K segment."""
    physical_pages = nl.ndarray(
        (1, pages_per_segment), dtype=block_table_sbuf.dtype, buffer=nl.sbuf
    )
    page_starts = nl.ndarray((1, pages_per_segment), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=physical_pages,
        src=block_table_sbuf[
            nl.ds(0, 1),
            nl.ds(
                seq_idx * max_num_pages + segment_idx * pages_per_segment,
                pages_per_segment,
            ),
        ],
    )
    nisa.tensor_scalar(
        dst=page_starts,
        data=page_offsets,
        op0=nl.add,
        operand0=segment_idx * segment_tokens,
    )
    page_live = nl.greater(nl.broadcast_to(seq_len_f32, page_starts.shape), page_starts)
    return nl.where(
        page_live, physical_pages, inactive_pages, dtype=block_table_sbuf.dtype
    )


def load_k_tile(
    k_cache,
    physical_pages,
    kv_head,
    num_kv_heads,
    d_head,
    page_size,
    pages_per_k_tile,
    k_tile_tokens,
):
    """Load one private K tile in matmul-ready [head_dim, tokens] order."""
    k_sbuf = nl.ndarray((d_head, k_tile_tokens), dtype=k_cache.dtype, buffer=nl.sbuf)
    nisa.memset(k_sbuf, 0.0)
    for page_in_tile in nl.affine_range(pages_per_k_tile):
        tile_offset = page_in_tile * page_size
        physical_page = physical_pages[nl.ds(0, 1), nl.ds(page_in_tile, 1)]
        nisa.dma_copy(
            dst=k_sbuf[0:d_head, nl.ds(tile_offset, page_size)],
            src=k_cache.ap(
                pattern=[
                    [page_size, d_head],
                    [1, page_size],
                ],
                offset=kv_head * d_head * page_size,
                scalar_offset=physical_page,
                indirect_dim=0,
            ),
            dge_mode=nisa.dge_mode.swdge,
            oob_mode=nisa.oob_mode.skip,
        )
    return k_sbuf


def load_k_tile_from_segment(
    k_cache,
    physical_pages,
    kv_head,
    k_tile_in_segment,
    num_kv_heads,
    d_head,
    page_size,
    pages_per_k_tile,
    k_tile_tokens,
):
    """Load one K tile from a segment-sized physical page table."""
    k_sbuf = nl.ndarray((d_head, k_tile_tokens), dtype=k_cache.dtype, buffer=nl.sbuf)
    nisa.memset(k_sbuf, 0.0)
    for page_in_tile in nl.affine_range(pages_per_k_tile):
        tile_offset = page_in_tile * page_size
        page_idx = k_tile_in_segment * pages_per_k_tile + page_in_tile
        physical_page = physical_pages[nl.ds(0, 1), nl.ds(page_idx, 1)]
        nisa.dma_copy(
            dst=k_sbuf[0:d_head, nl.ds(tile_offset, page_size)],
            src=k_cache.ap(
                pattern=[
                    [page_size, d_head],
                    [1, page_size],
                ],
                offset=kv_head * d_head * page_size,
                scalar_offset=physical_page,
                indirect_dim=0,
            ),
            dge_mode=nisa.dge_mode.swdge,
            oob_mode=nisa.oob_mode.skip,
        )
    return k_sbuf


def load_v_subtile(
    v_cache,
    physical_pages,
    kv_head,
    v_subtile_idx,
    num_kv_heads,
    d_head,
    page_size,
    pages_per_v_tile,
    v_tile_tokens,
):
    """Load one V subtile from the already-built K-tile page table."""
    v_sbuf = nl.ndarray((v_tile_tokens, d_head), dtype=v_cache.dtype, buffer=nl.sbuf)
    nisa.memset(v_sbuf, 0.0)
    for page_in_tile in nl.affine_range(pages_per_v_tile):
        page_idx = v_subtile_idx * pages_per_v_tile + page_in_tile
        tile_offset = page_in_tile * page_size
        physical_page = physical_pages[nl.ds(0, 1), nl.ds(page_idx, 1)]
        nisa.dma_copy(
            dst=v_sbuf[nl.ds(tile_offset, page_size), 0:d_head],
            src=v_cache.ap(
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
    return v_sbuf


def load_v_subtile_from_segment(
    v_cache,
    physical_pages,
    kv_head,
    k_tile_in_segment,
    v_subtile_idx,
    num_kv_heads,
    d_head,
    page_size,
    pages_per_k_tile,
    pages_per_v_tile,
    v_tile_tokens,
):
    """Load one V subtile from a segment-sized physical page table."""
    v_sbuf = nl.ndarray((v_tile_tokens, d_head), dtype=v_cache.dtype, buffer=nl.sbuf)
    nisa.memset(v_sbuf, 0.0)
    for page_in_tile in nl.affine_range(pages_per_v_tile):
        page_idx = (
            k_tile_in_segment * pages_per_k_tile
            + v_subtile_idx * pages_per_v_tile
            + page_in_tile
        )
        tile_offset = page_in_tile * page_size
        physical_page = physical_pages[nl.ds(0, 1), nl.ds(page_idx, 1)]
        nisa.dma_copy(
            dst=v_sbuf[nl.ds(tile_offset, page_size), 0:d_head],
            src=v_cache.ap(
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
    return v_sbuf


# Decode helpers and entrypoint.


def load_q_group(
    q_grouped, seq_idx, kv_head, num_kv_heads, heads_per_kv, d_head, scale
):
    """Load all GQA query heads for one (sequence, KV head).

    q is loaded as [heads_per_kv, d_head], transposed to
    [d_head, heads_per_kv], then one K/V stream serves all GQA heads.
    """
    q_row = seq_idx * num_kv_heads + kv_head
    q_sbuf = nl.load_transpose2d(
        q_grouped[q_row, 0:heads_per_kv, 0:d_head],
        dtype=q_grouped.dtype,
    )
    nisa.activation(dst=q_sbuf, op=nl.copy, data=q_sbuf, scale=scale)
    return q_sbuf


def update_decode_cache_inline(
    k_cache,
    v_cache,
    new_k,
    new_v,
    slot_mapping,
    num_tokens,
    num_kv_heads,
    d_head,
    page_size,
):
    """Write decode K/V rows into the paged cache before attention reads it.

    Decode attention includes the current token in its history. Writing the
    cache row here preserves that ordering before the K/V stream is loaded.

    Padded slot_mapping rows are allowed. The destination AP writes use
    oob_mode=skip, so invalid sentinel slots become no-op cache writes.
    """
    k_cache_private = k_cache.reshape(
        (k_cache.shape[0] * page_size * num_kv_heads * d_head, 1)
    )
    v_cache_flat = v_cache.reshape((v_cache.shape[0] * page_size, num_kv_heads, d_head))
    new_k_flat = new_k.reshape((num_tokens * num_kv_heads * d_head, 1))
    slot_mapping_2d = slot_mapping.reshape((1, num_tokens))

    for token_idx in nl.sequential_range(num_tokens):
        slot_sbuf = nl.ndarray((1, 1), dtype=slot_mapping.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=slot_sbuf,
            src=slot_mapping_2d[nl.ds(0, 1), nl.ds(token_idx, 1)],
        )
        k_private_offset = private_k_slot_offset(
            slot_sbuf,
            k_cache.shape[0],
            page_size,
            num_kv_heads,
            d_head,
        )

        for kv_head in nl.affine_range(num_kv_heads):
            k_sbuf = nl.ndarray((d_head, 1), dtype=new_k.dtype, buffer=nl.sbuf)
            v_sbuf = nl.ndarray((1, d_head), dtype=new_v.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=k_sbuf,
                src=new_k_flat.ap(
                    pattern=[[1, d_head], [1, 1]],
                    offset=(token_idx * num_kv_heads + kv_head) * d_head,
                ),
            )
            nisa.dma_copy(
                dst=v_sbuf,
                src=new_v[nl.ds(token_idx, 1), kv_head, 0:d_head],
            )
            nisa.dma_copy(
                dst=k_cache_private.ap(
                    pattern=[[page_size, d_head], [1, 1]],
                    offset=kv_head * d_head * page_size,
                    scalar_offset=k_private_offset,
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


def compute_decode_kv_head_group(
    q_grouped,
    out_grouped,
    k_cache,
    v_cache,
    block_table_sbuf,
    seq_len_f32,
    q_idx,
    kv_head,
    num_kv_heads,
    heads_per_kv,
    d_head,
    page_size,
    pages_per_k_tile,
    pages_per_v_tile,
    K_TILE,
    V_TILE,
    k_tiles_per_segment,
    num_k_segments,
    v_tiles_per_k_tile,
    segment_page_offsets,
    inactive_segment_pages,
    pages_per_segment,
    max_num_pages,
    scale,
    positions,
):
    """Compute one decode row for one KV head, sharing K/V across GQA heads.

    Decode intentionally has no register-controlled Q-block sweep: one batch
    lane enters this helper at a time. The segment/tile loops below are the
    K/V streaming online-softmax recurrence.
    """
    q_sbuf = load_q_group(
        q_grouped,
        q_idx,
        kv_head,
        num_kv_heads,
        heads_per_kv,
        d_head,
        scale,
    )

    qk_psum = nl.ndarray((heads_per_kv, K_TILE), dtype=nl.float32, buffer=nl.psum)
    qk_sbuf = nl.ndarray((heads_per_kv, K_TILE), dtype=nl.float32, buffer=nl.sbuf)
    masked_qk = nl.ndarray((heads_per_kv, K_TILE), dtype=nl.float32, buffer=nl.sbuf)
    neg_tile = nl.full(
        (heads_per_kv, K_TILE), NEG_INF, dtype=nl.float32, buffer=nl.sbuf
    )
    row_max = nl.ndarray((heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf)
    sum_row = nl.ndarray((heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf)
    running_out = nl.ndarray((heads_per_kv, d_head), dtype=nl.float32, buffer=nl.sbuf)
    tile_max = nl.ndarray((heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf)
    tile_sum = nl.ndarray((heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf)
    new_row_max = nl.ndarray((heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf)
    old_max_delta = nl.ndarray((heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf)
    old_max_scale = nl.ndarray((heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf)
    scaled_sum_row = nl.ndarray((heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf)
    new_sum_row = nl.ndarray((heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf)
    scaled_running_out = nl.ndarray(
        (heads_per_kv, d_head), dtype=nl.float32, buffer=nl.sbuf
    )
    exp_tile = nl.ndarray((heads_per_kv, K_TILE), dtype=nl.float32, buffer=nl.sbuf)
    neg_new_row_max = nl.ndarray((heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf)
    exp_t_psum = nl.ndarray((V_TILE, heads_per_kv), dtype=nl.float32, buffer=nl.psum)
    exp_t = nl.ndarray((V_TILE, heads_per_kv), dtype=v_cache.dtype, buffer=nl.sbuf)
    attn_out_psum = nl.ndarray((heads_per_kv, d_head), dtype=nl.float32, buffer=nl.psum)
    tile_out = nl.ndarray((heads_per_kv, d_head), dtype=nl.float32, buffer=nl.sbuf)
    inverse_sum_row = nl.ndarray((heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf)
    attn_out_sbuf = nl.ndarray(
        (heads_per_kv, d_head), dtype=out_grouped.dtype, buffer=nl.sbuf
    )

    nisa.memset(row_max, NEG_INF)
    nisa.memset(sum_row, 0.0)
    nisa.memset(running_out, 0.0)

    for segment_idx in nl.sequential_range(num_k_segments):
        physical_pages = build_k_segment_pages(
            block_table_sbuf,
            seq_len_f32,
            q_idx,
            segment_idx,
            segment_page_offsets,
            inactive_segment_pages,
            max_num_pages,
            pages_per_segment,
            pages_per_segment * page_size,
        )

        for k_tile_in_segment in nl.sequential_range(k_tiles_per_segment):
            k_tile_idx = segment_idx * k_tiles_per_segment + k_tile_in_segment
            k_sbuf = load_k_tile_from_segment(
                k_cache,
                physical_pages,
                kv_head,
                k_tile_in_segment,
                num_kv_heads,
                d_head,
                page_size,
                pages_per_k_tile,
                K_TILE,
            )

            nisa.memset(qk_psum, 0.0)
            nisa.nc_matmul(dst=qk_psum, stationary=q_sbuf, moving=k_sbuf)
            nisa.tensor_copy(dst=qk_sbuf, src=qk_psum)

            nisa.iota(positions, [[1, K_TILE]], offset=k_tile_idx * K_TILE)
            valid_seq = nl.less(
                positions,
                nl.broadcast_to(seq_len_f32, positions.shape),
            )
            masked_qk = nl.where(
                nl.broadcast_to(valid_seq, masked_qk.shape),
                qk_sbuf,
                neg_tile,
                dtype=nl.float32,
            )
            nisa.tensor_reduce(dst=tile_max, op=nl.maximum, data=masked_qk, axis=(1,))
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
            nisa.activation(dst=old_max_scale, op=nl.exp, data=old_max_delta)
            nisa.tensor_scalar(
                dst=neg_new_row_max,
                data=new_row_max,
                op0=nl.multiply,
                operand0=-1.0,
            )
            nisa.activation(
                dst=exp_tile,
                op=nl.exp,
                data=masked_qk,
                bias=neg_new_row_max,
                reduce_op=nl.add,
                reduce_res=tile_sum,
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
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

            nisa.memset(attn_out_psum, 0.0)
            for v_subtile_idx in nl.affine_range(v_tiles_per_k_tile):
                v_start = v_subtile_idx * V_TILE
                nisa.nc_transpose(
                    dst=exp_t_psum,
                    data=exp_tile[:, nl.ds(v_start, V_TILE)],
                )
                nisa.tensor_copy(dst=exp_t, src=exp_t_psum)
                v_sbuf = load_v_subtile_from_segment(
                    v_cache,
                    physical_pages,
                    kv_head,
                    k_tile_in_segment,
                    v_subtile_idx,
                    num_kv_heads,
                    d_head,
                    page_size,
                    pages_per_k_tile,
                    pages_per_v_tile,
                    V_TILE,
                )
                nisa.nc_matmul(dst=attn_out_psum, stationary=exp_t, moving=v_sbuf)
            nisa.tensor_copy(dst=tile_out, src=attn_out_psum)
            nisa.tensor_tensor(
                dst=running_out,
                data1=scaled_running_out,
                data2=tile_out,
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
    out_row = q_idx * num_kv_heads + kv_head
    nisa.dma_copy(
        dst=out_grouped[out_row, 0:heads_per_kv, 0:d_head],
        src=attn_out_sbuf,
    )


def paged_attention_decode_2d(
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
    """Decode paged attention with shared GQA K/V loads.

    The cache update runs first so the current decode row is visible in the
    subsequent attention stream. The loop order is batch lane -> KV head; for
    each KV head, one K/V tile stream feeds all query heads in its GQA group.
    Do not add the prefill q_loop_reg pattern here: decode has no compact
    Q-block sweep to hide from tracing.
    """
    num_tokens = q.shape[0]
    num_kv_heads = q.shape[1]
    heads_per_kv = q.shape[2]
    d_head = q.shape[3]
    page_size = k_cache.shape[1]
    max_num_pages = block_table.shape[1]
    batch_size = block_table.shape[0]
    scale = 1.0 / (d_head**0.5)
    k_tile_tokens = nl.tile_size.gemm_moving_fmax

    assert page_size == 16 or page_size == 32 or page_size == 64
    assert heads_per_kv <= nl.tile_size.pmax
    assert d_head == nl.tile_size.pmax
    assert num_tokens == batch_size
    assert k_cache.shape == v_cache.shape
    assert new_k.shape == new_v.shape
    assert new_k.shape[0] == num_tokens
    assert new_k.shape[1] == num_kv_heads
    assert new_k.shape[2] == d_head
    assert k_cache.shape[2] == num_kv_heads
    assert k_cache.shape[3] == d_head
    assert seq_lens.shape[0] == batch_size
    assert seq_lens.shape[1] == 1

    _ = query_start_len
    K_TILE = choose_page_aligned_tile(max_num_pages, page_size, k_tile_tokens)
    V_TILE = min(nl.tile_size.pmax, K_TILE)
    pages_per_k_tile = K_TILE // page_size
    pages_per_v_tile = V_TILE // page_size
    assert max_num_pages % pages_per_k_tile == 0
    num_k_tiles = max_num_pages // pages_per_k_tile
    k_tiles_per_segment = choose_k_tiles_per_segment(num_k_tiles, 4)
    num_k_segments = num_k_tiles // k_tiles_per_segment
    pages_per_segment = pages_per_k_tile * k_tiles_per_segment
    v_tiles_per_k_tile = K_TILE // V_TILE

    out = nl.ndarray(q.shape, dtype=q.dtype, buffer=nl.shared_hbm)
    q_grouped = q.reshape((num_tokens * num_kv_heads, heads_per_kv, d_head))
    out_grouped = out.reshape((num_tokens * num_kv_heads, heads_per_kv, d_head))

    update_decode_cache_inline(
        k_cache,
        v_cache,
        new_k,
        new_v,
        slot_mapping,
        num_tokens,
        num_kv_heads,
        d_head,
        page_size,
    )

    block_table_sbuf = nl.ndarray(
        (1, batch_size * max_num_pages), dtype=block_table.dtype, buffer=nl.sbuf
    )
    seq_lens_sbuf = nl.ndarray((1, batch_size), dtype=seq_lens.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=block_table_sbuf,
        src=block_table.reshape((1, batch_size * max_num_pages))[
            0:1, 0 : batch_size * max_num_pages
        ],
    )
    nisa.dma_copy(
        dst=seq_lens_sbuf,
        src=seq_lens.reshape((1, batch_size))[0:1, 0:batch_size],
    )
    segment_page_offsets = make_page_offsets(pages_per_segment, page_size)
    inactive_segment_pages = nl.full(
        (1, pages_per_segment), -1, dtype=block_table.dtype, buffer=nl.sbuf
    )
    positions = nl.ndarray((1, K_TILE), dtype=nl.float32, buffer=nl.sbuf)

    for q_idx in nl.sequential_range(num_tokens):
        seq_len = seq_lens_sbuf[nl.ds(0, 1), nl.ds(q_idx, 1)]
        seq_len_f32 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=seq_len_f32, src=seq_len)

        for kv_head in nl.sequential_range(num_kv_heads):
            compute_decode_kv_head_group(
                q_grouped,
                out_grouped,
                k_cache,
                v_cache,
                block_table_sbuf,
                seq_len_f32,
                q_idx,
                kv_head,
                num_kv_heads,
                heads_per_kv,
                d_head,
                page_size,
                pages_per_k_tile,
                pages_per_v_tile,
                K_TILE,
                V_TILE,
                k_tiles_per_segment,
                num_k_segments,
                v_tiles_per_k_tile,
                segment_page_offsets,
                inactive_segment_pages,
                pages_per_segment,
                max_num_pages,
                scale,
                positions,
            )

    return out


def load_grouped_q_tile(
    q_rows,
    local_q_start,
    kv_head,
    all_heads_dim,
    heads_per_kv,
    q_per_head_tile,
    q_group_tile,
    d_head,
    scale,
):
    """Load [heads_per_kv, q_per_head_tile] as one matmul partition tile.

    Query rows are laid out as:
      head_group 0 query rows, then head_group 1 query rows, ...

    After transpose, the tile is [d_head, heads_per_kv * q_per_head_tile], so
    one K/V stream can feed all query rows in the grouped tile.
    """
    q_loaded = nl.ndarray((q_group_tile, d_head), dtype=q_rows.dtype, buffer=nl.sbuf)
    q_transposed = nl.ndarray((d_head, q_group_tile), dtype=nl.float32, buffer=nl.psum)
    q_sbuf = nl.ndarray((d_head, q_group_tile), dtype=q_rows.dtype, buffer=nl.sbuf)
    nisa.memset(q_loaded, 0.0)
    for head_group in nl.affine_range(heads_per_kv):
        group_offset = head_group * q_per_head_tile
        head_offset = (kv_head * heads_per_kv + head_group) * d_head
        nisa.dma_copy(
            dst=q_loaded[nl.ds(group_offset, q_per_head_tile), 0:d_head],
            src=q_rows.ap(
                pattern=[[all_heads_dim, q_per_head_tile], [1, d_head]],
                offset=head_offset,
                scalar_offset=local_q_start,
                indirect_dim=0,
            ),
            dge_mode=nisa.dge_mode.swdge,
            oob_mode=nisa.oob_mode.skip,
        )
    nisa.nc_transpose(dst=q_transposed, data=q_loaded)
    nisa.tensor_copy(dst=q_sbuf, src=q_transposed)
    nisa.activation(dst=q_sbuf, op=nl.copy, data=q_sbuf, scale=scale)
    return q_sbuf


def build_grouped_q_metadata(
    local_q_start,
    context_tail,
    q_end_seq,
    heads_per_kv,
    q_per_head_tile,
    q_group_tile,
):
    """Build absolute query positions and live-row mask for grouped GQA lanes."""
    base_positions = nl.ndarray((q_per_head_tile, 1), dtype=nl.float32, buffer=nl.sbuf)
    q_rows = nl.ndarray((q_group_tile, 1), dtype=nl.float32, buffer=nl.sbuf)
    q_positions_abs = nl.ndarray((q_group_tile, 1), dtype=nl.float32, buffer=nl.sbuf)
    local_q_start_f32 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    q_end_f32 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(base_positions, [[q_per_head_tile, 1]], channel_multiplier=1)
    for head_group in nl.affine_range(heads_per_kv):
        group_offset = head_group * q_per_head_tile
        nisa.tensor_copy(
            dst=q_rows[nl.ds(group_offset, q_per_head_tile), 0:1],
            src=base_positions,
        )
    nisa.tensor_copy(dst=local_q_start_f32, src=local_q_start)
    nisa.tensor_copy(dst=q_end_f32, src=q_end_seq)
    nisa.tensor_tensor(
        dst=q_rows,
        data1=q_rows,
        data2=nl.broadcast_to(local_q_start_f32, q_rows.shape),
        op=nl.add,
    )
    valid_q_row = nl.less(q_rows, nl.broadcast_to(q_end_f32, q_rows.shape))
    nisa.tensor_tensor(
        dst=q_positions_abs,
        data1=q_rows,
        data2=nl.broadcast_to(context_tail, q_positions_abs.shape),
        op=nl.add,
    )
    return q_positions_abs, valid_q_row


def store_grouped_output(
    out_rows,
    attn_out_sbuf,
    local_q_start,
    kv_head,
    all_heads_dim,
    heads_per_kv,
    q_per_head_tile,
    d_head,
):
    """Scatter grouped prefill output rows back to compact [token, head] layout."""
    for head_group in nl.affine_range(heads_per_kv):
        group_offset = head_group * q_per_head_tile
        head_offset = (kv_head * heads_per_kv + head_group) * d_head
        nisa.dma_copy(
            dst=out_rows.ap(
                pattern=[[all_heads_dim, q_per_head_tile], [1, d_head]],
                offset=head_offset,
                scalar_offset=local_q_start,
                indirect_dim=0,
            ),
            src=attn_out_sbuf[nl.ds(group_offset, q_per_head_tile), 0:d_head],
            dge_mode=nisa.dge_mode.swdge,
            oob_mode=nisa.oob_mode.skip,
        )


def compute_prefill_q_block(
    q_rows,
    out_rows,
    k_cache,
    v_cache,
    block_table_sbuf,
    seq_lens_sbuf,
    seq_idx,
    kv_head,
    current_q_start,
    q_end_seq,
    all_heads_dim,
    num_kv_heads,
    heads_per_kv,
    q_per_head_tile,
    q_group_tile,
    d_head,
    page_size,
    pages_per_k_tile,
    pages_per_v_tile,
    K_TILE,
    V_TILE,
    num_k_tiles,
    v_tiles_per_k_tile,
    page_offsets,
    inactive_pages,
    max_num_pages,
    scale,
    key_positions_f32,
):
    """Compute one prefill `(sequence, q-block, kv-head)` tile.

    This helper keeps the per-Q-block body shared by the serial prefill loop;
    the only loop-carried recurrence inside it is the K/V streaming softmax state.
    """
    seq_len = seq_lens_sbuf[nl.ds(0, 1), nl.ds(seq_idx, 1)]
    seq_len_f32 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    q_end_f32 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    context_tail = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=seq_len_f32, src=seq_len)
    nisa.tensor_copy(dst=q_end_f32, src=q_end_seq)
    nisa.tensor_scalar(
        dst=context_tail,
        data=seq_len_f32,
        op0=nl.subtract,
        operand0=q_end_f32,
    )

    q_sbuf = load_grouped_q_tile(
        q_rows,
        current_q_start,
        kv_head,
        all_heads_dim,
        heads_per_kv,
        q_per_head_tile,
        q_group_tile,
        d_head,
        scale,
    )
    q_positions_abs, valid_q_row = build_grouped_q_metadata(
        current_q_start,
        context_tail,
        q_end_seq,
        heads_per_kv,
        q_per_head_tile,
        q_group_tile,
    )

    qk_psum = nl.ndarray((q_group_tile, K_TILE), dtype=nl.float32, buffer=nl.psum)
    qk_sbuf = nl.ndarray((q_group_tile, K_TILE), dtype=nl.float32, buffer=nl.sbuf)
    masked_qk = nl.ndarray((q_group_tile, K_TILE), dtype=nl.float32, buffer=nl.sbuf)
    neg_tile = nl.full(
        (q_group_tile, K_TILE), NEG_INF, dtype=nl.float32, buffer=nl.sbuf
    )
    row_max = nl.ndarray((q_group_tile, 1), dtype=nl.float32, buffer=nl.sbuf)
    sum_row = nl.ndarray((q_group_tile, 1), dtype=nl.float32, buffer=nl.sbuf)
    running_out = nl.ndarray((q_group_tile, d_head), dtype=nl.float32, buffer=nl.sbuf)
    tile_max = nl.ndarray((q_group_tile, 1), dtype=nl.float32, buffer=nl.sbuf)
    tile_sum = nl.ndarray((q_group_tile, 1), dtype=nl.float32, buffer=nl.sbuf)
    new_row_max = nl.ndarray((q_group_tile, 1), dtype=nl.float32, buffer=nl.sbuf)
    old_max_delta = nl.ndarray((q_group_tile, 1), dtype=nl.float32, buffer=nl.sbuf)
    old_max_scale = nl.ndarray((q_group_tile, 1), dtype=nl.float32, buffer=nl.sbuf)
    scaled_sum_row = nl.ndarray((q_group_tile, 1), dtype=nl.float32, buffer=nl.sbuf)
    new_sum_row = nl.ndarray((q_group_tile, 1), dtype=nl.float32, buffer=nl.sbuf)
    scaled_running_out = nl.ndarray(
        (q_group_tile, d_head), dtype=nl.float32, buffer=nl.sbuf
    )
    exp_tile = nl.ndarray((q_group_tile, K_TILE), dtype=nl.float32, buffer=nl.sbuf)
    neg_new_row_max = nl.ndarray((q_group_tile, 1), dtype=nl.float32, buffer=nl.sbuf)
    exp_t_psum = nl.ndarray((V_TILE, q_group_tile), dtype=nl.float32, buffer=nl.psum)
    exp_t = nl.ndarray((V_TILE, q_group_tile), dtype=v_cache.dtype, buffer=nl.sbuf)
    attn_out_psum = nl.ndarray(
        (q_group_tile, d_head), dtype=nl.float32, buffer=nl.psum
    )
    tile_out = nl.ndarray((q_group_tile, d_head), dtype=nl.float32, buffer=nl.sbuf)

    one_row = nl.full((q_group_tile, 1), 1.0, dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(row_max, NEG_INF)
    row_max = nl.where(valid_q_row, row_max, one_row, dtype=nl.float32)
    nisa.memset(sum_row, 0.0)
    nisa.memset(running_out, 0.0)

    for k_tile_idx in nl.sequential_range(num_k_tiles):
        physical_pages = build_k_tile_pages(
            block_table_sbuf,
            seq_len_f32,
            seq_idx,
            k_tile_idx,
            page_offsets,
            inactive_pages,
            max_num_pages,
            pages_per_k_tile,
            K_TILE,
        )
        k_sbuf = load_k_tile(
            k_cache,
            physical_pages,
            kv_head,
            num_kv_heads,
            d_head,
            page_size,
            pages_per_k_tile,
            K_TILE,
        )
        nisa.memset(qk_psum, 0.0)
        nisa.nc_matmul(dst=qk_psum, stationary=q_sbuf, moving=k_sbuf)
        nisa.tensor_copy(dst=qk_sbuf, src=qk_psum)

        nisa.iota(key_positions_f32, [[1, K_TILE]], offset=k_tile_idx * K_TILE)
        causal_mask = nl.greater_equal(
            nl.broadcast_to(q_positions_abs, qk_sbuf.shape),
            nl.broadcast_to(key_positions_f32, qk_sbuf.shape),
        )
        combined_mask = nl.logical_and(
            causal_mask,
            nl.broadcast_to(valid_q_row, qk_sbuf.shape),
        )
        masked_qk = nl.where(combined_mask, qk_sbuf, neg_tile, dtype=nl.float32)
        nisa.tensor_reduce(dst=tile_max, op=nl.maximum, data=masked_qk, axis=(1,))
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
        nisa.activation(dst=old_max_scale, op=nl.exp, data=old_max_delta)
        nisa.tensor_scalar(
            dst=neg_new_row_max,
            data=new_row_max,
            op0=nl.multiply,
            operand0=-1.0,
        )
        nisa.activation(
            dst=exp_tile,
            op=nl.exp,
            data=masked_qk,
            bias=neg_new_row_max,
            reduce_op=nl.add,
            reduce_res=tile_sum,
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
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
        nisa.memset(attn_out_psum, 0.0)
        for v_subtile_idx in nl.affine_range(v_tiles_per_k_tile):
            v_start = v_subtile_idx * V_TILE
            nisa.nc_transpose(
                dst=exp_t_psum,
                data=exp_tile[:, nl.ds(v_start, V_TILE)],
            )
            nisa.tensor_copy(dst=exp_t, src=exp_t_psum)
            v_sbuf = load_v_subtile(
                v_cache,
                physical_pages,
                kv_head,
                v_subtile_idx,
                num_kv_heads,
                d_head,
                page_size,
                pages_per_v_tile,
                V_TILE,
            )
            nisa.nc_matmul(dst=attn_out_psum, stationary=exp_t, moving=v_sbuf)
        nisa.tensor_copy(dst=tile_out, src=attn_out_psum)
        nisa.tensor_tensor(
            dst=running_out,
            data1=scaled_running_out,
            data2=tile_out,
            op=nl.add,
        )
        nisa.tensor_copy(dst=row_max, src=new_row_max)
        nisa.tensor_copy(dst=sum_row, src=new_sum_row)

    inverse_sum_row = nl.ndarray((q_group_tile, 1), dtype=nl.float32, buffer=nl.sbuf)
    zero_out = nl.full((q_group_tile, d_head), 0.0, dtype=nl.float32, buffer=nl.sbuf)
    safe_sum_row = nl.where(valid_q_row, sum_row, one_row, dtype=nl.float32)
    safe_running_out = nl.where(
        nl.broadcast_to(valid_q_row, running_out.shape),
        running_out,
        zero_out,
        dtype=nl.float32,
    )
    nisa.reciprocal(dst=inverse_sum_row, data=safe_sum_row)
    nisa.tensor_scalar(
        dst=safe_running_out,
        data=safe_running_out,
        op0=nl.multiply,
        operand0=inverse_sum_row,
    )
    attn_out_sbuf = nl.ndarray((q_group_tile, d_head), dtype=v_cache.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=attn_out_sbuf, src=safe_running_out)
    store_grouped_output(
        out_rows,
        attn_out_sbuf,
        current_q_start,
        kv_head,
        all_heads_dim,
        heads_per_kv,
        q_per_head_tile,
        d_head,
    )


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
    """Ragged prefill/mixed paged attention with GQA Q grouping.

    q is stored as compact rows, and K/V history is addressed by sequence via
    block_table. query_start_len gives each sequence's live q-row interval. For
    each sequence and KV head, the kernel walks that interval in grouped Q
    tiles, streams physical K/V pages, applies the causal mask from absolute
    query/key positions, and scatters the grouped result back to compact rows.

    Grouping places multiple GQA query heads in the same tile so they share the
    same page metadata and K/V loads for a KV head.
    """
    num_tokens = q.shape[0]
    num_kv_heads = q.shape[1]
    heads_per_kv = q.shape[2]
    d_head = q.shape[3]
    page_size = k_cache.shape[1]
    max_num_pages = block_table.shape[1]
    batch_size = block_table.shape[0]
    scale = 1.0 / (d_head**0.5)

    assert page_size == 16 or page_size == 32 or page_size == 64
    assert d_head == nl.tile_size.pmax
    assert heads_per_kv <= nl.tile_size.pmax
    assert nl.tile_size.pmax % heads_per_kv == 0
    assert q.shape[0] >= batch_size
    assert k_cache.shape == v_cache.shape
    assert new_k.shape == new_v.shape
    assert new_k.shape[0] == num_tokens
    assert new_k.shape[1] == num_kv_heads
    assert new_k.shape[2] == d_head
    assert k_cache.shape[2] == num_kv_heads
    assert k_cache.shape[3] == d_head
    assert seq_lens.shape[0] == batch_size
    assert seq_lens.shape[1] == 1
    assert query_start_len.shape[0] == batch_size + 1
    assert query_start_len.shape[1] == 1

    K_TILE = choose_page_aligned_tile(
        max_num_pages, page_size, nl.tile_size.gemm_moving_fmax
    )
    V_TILE = min(nl.tile_size.pmax, K_TILE)
    q_per_head_tile = nl.tile_size.pmax // heads_per_kv
    q_group_tile = q_per_head_tile * heads_per_kv
    pages_per_k_tile = K_TILE // page_size
    pages_per_v_tile = V_TILE // page_size
    assert max_num_pages % pages_per_k_tile == 0
    num_k_tiles = max_num_pages // pages_per_k_tile
    v_tiles_per_k_tile = K_TILE // V_TILE

    out = nl.ndarray(q.shape, dtype=q.dtype, buffer=nl.shared_hbm)
    all_heads_dim = num_kv_heads * heads_per_kv * d_head
    q_rows = q.reshape((num_tokens, all_heads_dim))
    out_rows = out.reshape((num_tokens, all_heads_dim))

    block_table_sbuf, seq_lens_sbuf, query_start_len_sbuf = (
        load_static_attention_metadata(
            block_table,
            seq_lens,
            query_start_len,
            batch_size,
            max_num_pages,
        )
    )
    page_offsets = make_page_offsets(pages_per_k_tile, page_size)
    inactive_pages = nl.full(
        (1, pages_per_k_tile), -1, dtype=block_table.dtype, buffer=nl.sbuf
    )
    key_positions_f32 = nl.ndarray((1, K_TILE), dtype=nl.float32, buffer=nl.sbuf)

    for kv_head in nl.affine_range(num_kv_heads):
        for seq_idx in nl.affine_range(batch_size):
            q_start_seq = query_start_len_sbuf[nl.ds(0, 1), nl.ds(seq_idx, 1)]
            q_end_seq = query_start_len_sbuf[nl.ds(0, 1), nl.ds(seq_idx + 1, 1)]
            seq_len = seq_lens_sbuf[nl.ds(0, 1), nl.ds(seq_idx, 1)]

            current_q_start = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            q_start_f32 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            q_end_f32 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            q_tokens_remaining = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            loop_one_i32 = nl.full((1, 1), 1, dtype=nl.int32, buffer=nl.sbuf)
            loop_zero_i32 = nl.full((1, 1), 0, dtype=nl.int32, buffer=nl.sbuf)
            loop_zero_f32 = nl.full((1, 1), 0.0, dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=current_q_start, src=q_start_seq)
            nisa.tensor_copy(dst=q_start_f32, src=q_start_seq)
            nisa.tensor_copy(dst=q_end_f32, src=q_end_seq)
            nisa.tensor_scalar(
                dst=q_tokens_remaining,
                data=q_end_f32,
                op0=nl.subtract,
                operand0=q_start_f32,
            )
            q_loop_live = nl.greater(q_tokens_remaining, loop_zero_f32)
            q_loop_continue = nl.where(
                q_loop_live, loop_one_i32, loop_zero_i32, dtype=nl.int32
            )
            # Keep this prefill-only register loop unless grid/SPMD or graph
            # chunking replaces it; it prevents tracing one body per Q block.
            q_loop_reg = nisa.register_alloc(None)
            nisa.register_load(dst=q_loop_reg, src=q_loop_continue)

            while q_loop_reg:
                local_q_start = nisa.register_alloc(None)
                nisa.register_load(dst=local_q_start, src=current_q_start)

                seq_len_f32 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
                context_tail = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=seq_len_f32, src=seq_len)
                nisa.tensor_scalar(
                    dst=context_tail,
                    data=seq_len_f32,
                    op0=nl.subtract,
                    operand0=q_end_f32,
                )

                q_sbuf = load_grouped_q_tile(
                    q_rows,
                    local_q_start,
                    kv_head,
                    all_heads_dim,
                    heads_per_kv,
                    q_per_head_tile,
                    q_group_tile,
                    d_head,
                    scale,
                )
                q_positions_abs, valid_q_row = build_grouped_q_metadata(
                    current_q_start,
                    context_tail,
                    q_end_seq,
                    heads_per_kv,
                    q_per_head_tile,
                    q_group_tile,
                )

                qk_psum = nl.ndarray(
                    (q_group_tile, K_TILE), dtype=nl.float32, buffer=nl.psum
                )
                qk_sbuf = nl.ndarray(
                    (q_group_tile, K_TILE), dtype=nl.float32, buffer=nl.sbuf
                )
                masked_qk = nl.ndarray(
                    (q_group_tile, K_TILE), dtype=nl.float32, buffer=nl.sbuf
                )
                neg_tile = nl.full(
                    (q_group_tile, K_TILE),
                    NEG_INF,
                    dtype=nl.float32,
                    buffer=nl.sbuf,
                )
                row_max = nl.ndarray(
                    (q_group_tile, 1), dtype=nl.float32, buffer=nl.sbuf
                )
                sum_row = nl.ndarray(
                    (q_group_tile, 1), dtype=nl.float32, buffer=nl.sbuf
                )
                running_out = nl.ndarray(
                    (q_group_tile, d_head), dtype=nl.float32, buffer=nl.sbuf
                )
                tile_max = nl.ndarray(
                    (q_group_tile, 1), dtype=nl.float32, buffer=nl.sbuf
                )
                tile_sum = nl.ndarray(
                    (q_group_tile, 1), dtype=nl.float32, buffer=nl.sbuf
                )
                new_row_max = nl.ndarray(
                    (q_group_tile, 1), dtype=nl.float32, buffer=nl.sbuf
                )
                old_max_delta = nl.ndarray(
                    (q_group_tile, 1), dtype=nl.float32, buffer=nl.sbuf
                )
                old_max_scale = nl.ndarray(
                    (q_group_tile, 1), dtype=nl.float32, buffer=nl.sbuf
                )
                scaled_sum_row = nl.ndarray(
                    (q_group_tile, 1), dtype=nl.float32, buffer=nl.sbuf
                )
                new_sum_row = nl.ndarray(
                    (q_group_tile, 1), dtype=nl.float32, buffer=nl.sbuf
                )
                scaled_running_out = nl.ndarray(
                    (q_group_tile, d_head), dtype=nl.float32, buffer=nl.sbuf
                )
                exp_tile = nl.ndarray(
                    (q_group_tile, K_TILE), dtype=nl.float32, buffer=nl.sbuf
                )
                neg_new_row_max = nl.ndarray(
                    (q_group_tile, 1), dtype=nl.float32, buffer=nl.sbuf
                )
                exp_t_psum = nl.ndarray(
                    (V_TILE, q_group_tile), dtype=nl.float32, buffer=nl.psum
                )
                exp_t = nl.ndarray(
                    (V_TILE, q_group_tile), dtype=v_cache.dtype, buffer=nl.sbuf
                )
                attn_out_psum = nl.ndarray(
                    (q_group_tile, d_head), dtype=nl.float32, buffer=nl.psum
                )
                tile_out = nl.ndarray(
                    (q_group_tile, d_head), dtype=nl.float32, buffer=nl.sbuf
                )

                one_row = nl.full(
                    (q_group_tile, 1), 1.0, dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.memset(row_max, NEG_INF)
                row_max = nl.where(valid_q_row, row_max, one_row, dtype=nl.float32)
                nisa.memset(sum_row, 0.0)
                nisa.memset(running_out, 0.0)

                for k_tile_idx in nl.sequential_range(num_k_tiles):
                    physical_pages = build_k_tile_pages(
                        block_table_sbuf,
                        seq_len_f32,
                        seq_idx,
                        k_tile_idx,
                        page_offsets,
                        inactive_pages,
                        max_num_pages,
                        pages_per_k_tile,
                        K_TILE,
                    )
                    k_sbuf = load_k_tile(
                        k_cache,
                        physical_pages,
                        kv_head,
                        num_kv_heads,
                        d_head,
                        page_size,
                        pages_per_k_tile,
                        K_TILE,
                    )
                    nisa.memset(qk_psum, 0.0)
                    nisa.nc_matmul(dst=qk_psum, stationary=q_sbuf, moving=k_sbuf)
                    nisa.tensor_copy(dst=qk_sbuf, src=qk_psum)

                    nisa.iota(
                        key_positions_f32, [[1, K_TILE]], offset=k_tile_idx * K_TILE
                    )
                    causal_mask = nl.greater_equal(
                        nl.broadcast_to(q_positions_abs, qk_sbuf.shape),
                        nl.broadcast_to(key_positions_f32, qk_sbuf.shape),
                    )
                    combined_mask = nl.logical_and(
                        causal_mask,
                        nl.broadcast_to(valid_q_row, qk_sbuf.shape),
                    )
                    masked_qk = nl.where(
                        combined_mask,
                        qk_sbuf,
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
                    nisa.activation(dst=old_max_scale, op=nl.exp, data=old_max_delta)
                    nisa.tensor_scalar(
                        dst=neg_new_row_max,
                        data=new_row_max,
                        op0=nl.multiply,
                        operand0=-1.0,
                    )
                    nisa.activation(
                        dst=exp_tile,
                        op=nl.exp,
                        data=masked_qk,
                        bias=neg_new_row_max,
                        reduce_op=nl.add,
                        reduce_res=tile_sum,
                        reduce_cmd=nisa.reduce_cmd.reset_reduce,
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
                    nisa.memset(attn_out_psum, 0.0)
                    for v_subtile_idx in nl.affine_range(v_tiles_per_k_tile):
                        v_start = v_subtile_idx * V_TILE
                        nisa.nc_transpose(
                            dst=exp_t_psum,
                            data=exp_tile[:, nl.ds(v_start, V_TILE)],
                        )
                        nisa.tensor_copy(dst=exp_t, src=exp_t_psum)
                        v_sbuf = load_v_subtile(
                            v_cache,
                            physical_pages,
                            kv_head,
                            v_subtile_idx,
                            num_kv_heads,
                            d_head,
                            page_size,
                            pages_per_v_tile,
                            V_TILE,
                        )
                        nisa.nc_matmul(
                            dst=attn_out_psum, stationary=exp_t, moving=v_sbuf
                        )
                    nisa.tensor_copy(dst=tile_out, src=attn_out_psum)
                    nisa.tensor_tensor(
                        dst=running_out,
                        data1=scaled_running_out,
                        data2=tile_out,
                        op=nl.add,
                    )
                    nisa.tensor_copy(dst=row_max, src=new_row_max)
                    nisa.tensor_copy(dst=sum_row, src=new_sum_row)

                inverse_sum_row = nl.ndarray(
                    (q_group_tile, 1), dtype=nl.float32, buffer=nl.sbuf
                )
                zero_out = nl.full(
                    (q_group_tile, d_head), 0.0, dtype=nl.float32, buffer=nl.sbuf
                )
                safe_sum_row = nl.where(valid_q_row, sum_row, one_row, dtype=nl.float32)
                safe_running_out = nl.where(
                    nl.broadcast_to(valid_q_row, running_out.shape),
                    running_out,
                    zero_out,
                    dtype=nl.float32,
                )
                nisa.reciprocal(dst=inverse_sum_row, data=safe_sum_row)
                nisa.tensor_scalar(
                    dst=safe_running_out,
                    data=safe_running_out,
                    op0=nl.multiply,
                    operand0=inverse_sum_row,
                )
                attn_out_sbuf = nl.ndarray(
                    (q_group_tile, d_head), dtype=q.dtype, buffer=nl.sbuf
                )
                nisa.tensor_copy(dst=attn_out_sbuf, src=safe_running_out)
                store_grouped_output(
                    out_rows,
                    attn_out_sbuf,
                    local_q_start,
                    kv_head,
                    all_heads_dim,
                    heads_per_kv,
                    q_per_head_tile,
                    d_head,
                )

                nisa.tensor_scalar(
                    dst=current_q_start,
                    data=current_q_start,
                    op0=nl.add,
                    operand0=q_per_head_tile,
                )
                nisa.tensor_scalar(
                    dst=q_tokens_remaining,
                    data=q_tokens_remaining,
                    op0=nl.subtract,
                    operand0=float(q_per_head_tile),
                )
                q_loop_live = nl.greater(q_tokens_remaining, loop_zero_f32)
                q_loop_continue = nl.where(
                    q_loop_live, loop_one_i32, loop_zero_i32, dtype=nl.int32
                )
                nisa.register_load(dst=q_loop_reg, src=q_loop_continue)

    return out



# Standalone prefill cache-update entrypoint.


def paged_kv_cache_update(
    k_cache: nt.tensor,
    v_cache: nt.tensor,
    new_k,
    new_v,
    slot_mapping,
    query_start_len,
):
    """Scatter compact K/V update rows into the paged cache.

    Shapes:
      k_cache:         private [num_pages, num_kv_heads, head_dim, page_size]
                       physical order, carried in the public cache allocation
      v_cache:         [num_pages, page_size, num_kv_heads, head_dim]
      new_k/new_v:     [compiled_tokens, num_kv_heads, head_dim]
      slot_mapping:    [compiled_tokens]
      query_start_len: [batch_size + 1, 1]

    Each update row is copied through a small SBUF tile and written to the
    flattened cache slot named by slot_mapping. Padded rows use out-of-range
    slot values, so the AP destination write becomes a no-op under
    oob_mode=skip.
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
    k_cache_private = k_cache.reshape(
        (num_pages * page_size * num_kv_heads * d_head, 1)
    )
    v_cache_flat = v_cache.reshape((num_pages * page_size, num_kv_heads, d_head))
    new_k_flat = new_k.reshape((compiled_tokens * num_kv_heads * d_head, 1))
    slot_mapping_2d = slot_mapping.reshape((compiled_tokens, 1))

    for token_idx in nl.sequential_range(compiled_tokens):
        slot_sbuf = nl.ndarray((1, 1), dtype=slot_mapping.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=slot_sbuf,
            src=slot_mapping_2d[nl.ds(token_idx, 1), :],
        )
        k_private_offset = private_k_slot_offset(
            slot_sbuf,
            num_pages,
            page_size,
            num_kv_heads,
            d_head,
        )
        for kv_head in nl.affine_range(num_kv_heads):
            k_sbuf = nl.ndarray((d_head, 1), dtype=new_k.dtype, buffer=nl.sbuf)
            v_sbuf = nl.ndarray((1, d_head), dtype=new_v.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=k_sbuf,
                src=new_k_flat.ap(
                    pattern=[[1, d_head], [1, 1]],
                    offset=(token_idx * num_kv_heads + kv_head) * d_head,
                ),
            )
            nisa.dma_copy(
                dst=v_sbuf,
                src=new_v[nl.ds(token_idx, 1), kv_head, :],
            )
            nisa.dma_copy(
                dst=k_cache_private.ap(
                    pattern=[[page_size, d_head], [1, 1]],
                    offset=kv_head * d_head * page_size,
                    scalar_offset=k_private_offset,
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
