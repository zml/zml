"""Grouped matmul Mosaic-TPU harness adapter.

This file is a near-faithful port of MaxText's `megablox.backend.gmm`
kernel path for harness lowering/diffing.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from typing import Any, Dict, Optional, Tuple

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

__all__ = ["gmm", "megablox_gmm_kernel", "build_args"]


def _validate_args(
    *,
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    expected_rhs_dims: int = 3,
) -> jnp.ndarray:
    """Validates arguments for grouped matmul."""
    if lhs.ndim != 2:
        raise ValueError(f"Expected 2-tensor for 'lhs' but got {lhs.ndim}-tensor.")
    if rhs.ndim != expected_rhs_dims:
        raise ValueError(f"Expected {expected_rhs_dims}-tensor for 'rhs' but got {rhs.ndim}-tensor.")
    if group_sizes.dtype != jnp.int32:
        raise ValueError(f"Expected 32-bit integer 'group_sizes' but got {group_sizes.dtype}.")
    return group_sizes


def _calculate_num_tiles(x: int, tx: int) -> int:
    tiles, rem = divmod(x, tx)
    if rem:
        raise ValueError(f"{x} must be divisible by x-dimension tile size ({tx}).")
    return tiles


def _calculate_irregular_num_tiles(x: int, tx: int) -> tuple[int, int]:
    tiles, rem = divmod(x, tx)
    if rem:
        tiles += 1
    return tiles, rem


GroupMetadata = Any
LutFn = Callable[[int, int, int], Optional[tuple[int, int, int]]]


def make_group_metadata(
    *,
    group_sizes: jnp.ndarray,
    m: int,
    tm: int,
    start_group: jnp.ndarray,
    num_nonzero_groups: int,
    visit_empty_groups: bool = True,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Create grouped-matmul tile ownership metadata."""
    num_groups = group_sizes.shape[0]
    end_group = start_group + num_nonzero_groups - 1

    group_ends = jnp.cumsum(group_sizes)
    group_offsets = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), group_ends])

    rounded_group_ends = ((group_ends + tm - 1) // tm * tm).astype(jnp.int32)
    group_starts = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), group_ends[:-1]])
    rounded_group_starts = group_starts // tm * tm

    rounded_group_sizes = rounded_group_ends - rounded_group_starts
    rounded_group_sizes = jnp.where(group_sizes == 0, 0, rounded_group_sizes)
    group_tiles = rounded_group_sizes // tm

    if visit_empty_groups:
        group_tiles = jnp.where(group_sizes == 0, 1, group_tiles)

    tiles_m = _calculate_num_tiles(m, tm)
    group_ids = jnp.repeat(
        jnp.arange(num_groups, dtype=jnp.int32),
        group_tiles,
        total_repeat_length=tiles_m + num_groups - 1,
    )

    partial_tile_mask = jnp.logical_or((group_offsets[:-1] % tm) == 0, group_sizes == 0)
    if visit_empty_groups:
        partial_tile_mask = jnp.where(group_sizes == 0, 0, partial_tile_mask)
    partial_tile_ids = jnp.where(partial_tile_mask, tiles_m, group_offsets[:-1] // tm)
    tile_visits = jnp.histogram(partial_tile_ids, bins=tiles_m, range=(0, tiles_m - 1))[0] + 1

    m_tile_ids = jnp.repeat(
        jnp.arange(tiles_m, dtype=jnp.int32),
        tile_visits.astype(jnp.int32),
        total_repeat_length=tiles_m + num_groups - 1,
    )

    first_tile_in_shard = (group_ids < start_group).sum()
    group_ids = jnp.roll(group_ids, shift=-first_tile_in_shard, axis=0)
    m_tile_ids = jnp.roll(m_tile_ids, shift=-first_tile_in_shard, axis=0)

    iota = jnp.arange(num_groups, dtype=jnp.int32)
    active_group_mask = jnp.logical_and(iota <= end_group, iota >= start_group)
    group_tiles = jnp.where(active_group_mask, group_tiles, 0)
    num_tiles = group_tiles.sum()
    return (group_offsets, group_ids, m_tile_ids), num_tiles


def _zero_uninitialized_memory(
    out: jnp.ndarray,
    *,
    start_group: jnp.ndarray,
    num_nonzero_groups: int,
    group_metadata: GroupMetadata,
) -> jnp.ndarray:
    """Zero rows outside the active group shard."""
    group_offsets = group_metadata[0]
    group_start = group_offsets[start_group]
    group_end = group_offsets[start_group + num_nonzero_groups]
    valid_mask = jax.lax.broadcasted_iota(jnp.int32, (out.shape[0],), 0)
    valid_mask = (valid_mask >= group_start) & (valid_mask < group_end)
    return jnp.where(valid_mask[:, None], out, 0)


def _calculate_bytes(x: jax.Array) -> int:
    total_bytes = 0
    for leaf in jax.tree.leaves(x):
        total_bytes += leaf.dtype.itemsize * leaf.size
    return total_bytes


def _get_store_mask(
    *,
    grid_id: jnp.ndarray,
    group_metadata: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tm: int,
    tn: int,
) -> jnp.ndarray:
    group_offsets, group_ids, m_tile_ids = group_metadata
    group_id = group_ids[grid_id]
    group_start = group_offsets[group_id]
    group_end = group_offsets[group_id + 1]
    m_id = m_tile_ids[grid_id] * tm
    iota = jax.lax.broadcasted_iota(jnp.int32, (tm, tn), 0) + m_id
    return jnp.logical_and(iota >= group_start, iota < group_end)


@functools.partial(
    jax.jit,
    static_argnames=[
        "preferred_element_type",
        "tiling",
        "transpose_rhs",
        "interpret",
    ],
)
def gmm(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype = jnp.float32,
    tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128),
    group_offset: jnp.ndarray | None = None,
    existing_out: jnp.ndarray | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
) -> jnp.ndarray:
    """Compute lhs[sizes[i-1]:sizes[i], :] @ rhs for each group i."""
    if existing_out is not None:
        assert isinstance(existing_out, jax.Array)
        expected_dtype = existing_out.dtype
        if expected_dtype != preferred_element_type:
            raise ValueError("Existing output dtype must match preferred_element_type.")

    if group_offset is None:
        group_offset = jnp.array([0], dtype=jnp.int32)
    else:
        if group_offset.shape:
            raise ValueError(f"group_offset must be scalar; got {group_offset.shape}")
        group_offset = group_offset[None]

    num_current_groups = rhs.shape[0]
    num_total_groups = group_sizes.shape[0]
    group_sizes = _validate_args(lhs=lhs, rhs=rhs, group_sizes=group_sizes)

    m, k, n = (lhs.shape[0], lhs.shape[1], rhs.shape[2])
    if transpose_rhs:
        n = rhs.shape[1]

    if callable(tiling):
        tiling = tiling(m, k, n)
    if tiling is None:
        raise ValueError(f"No tuned tiling found for (m, k, n) = ({m}, {k}, {n})")

    tm, tk, tn = tiling
    tiles_k, k_rem = _calculate_irregular_num_tiles(k, tk)
    tiles_n, n_rem = _calculate_irregular_num_tiles(n, tn)
    del n_rem

    group_metadata, num_active_tiles = make_group_metadata(
        group_sizes=group_sizes,
        m=m,
        tm=tm,
        start_group=group_offset[0],
        num_nonzero_groups=rhs.shape[0],
        visit_empty_groups=False,
    )

    def kernel(
        group_metadata,
        group_offset,
        lhs: jax.Array,
        rhs: jax.Array,
        existing_out,
        out,
        acc_scratch,
    ):
        group_offsets, group_ids, m_tile_ids = group_metadata
        del group_offsets, group_ids, group_offset

        grid_id = pl.program_id(1)
        k_i = pl.program_id(2)

        @pl.when(k_i == 0)
        def _zero_acc():
            acc_scratch[...] = jnp.zeros_like(acc_scratch)

            if existing_out is not None:
                prev_grid_id = jnp.where(grid_id > 0, grid_id - 1, 0)
                is_first_processed_group = grid_id == 0
                m_tile_changed = m_tile_ids[grid_id] != m_tile_ids[prev_grid_id]
                first_time_seeing_out = jnp.logical_or(is_first_processed_group, m_tile_changed)

                @pl.when(first_time_seeing_out)
                def _init_out():
                    out[...] = existing_out[...]

        def mask_k_rem(x: jax.Array, *, dim: int):
            if k_rem == 0:
                return x
            iota = lax.broadcasted_iota(jnp.int32, x.shape, dim)
            return jnp.where(iota < k_rem, x, 0).astype(x.dtype)

        def _store_accum():
            mask = _get_store_mask(
                grid_id=grid_id,
                group_metadata=group_metadata,
                tm=tm,
                tn=tn,
            )
            out[...] = jax.lax.select(
                mask[...],
                acc_scratch[...],
                out[...].astype(jnp.float32),
            ).astype(preferred_element_type)

        def _accum(is_last_k_tile):
            if is_last_k_tile:
                mask_k_rem_lhs = functools.partial(mask_k_rem, dim=1)
                mask_k_rem_rhs = functools.partial(mask_k_rem, dim=int(transpose_rhs))
            else:
                mask_k_rem_lhs = lambda x: x
                mask_k_rem_rhs = lambda x: x

            loaded_lhs = mask_k_rem_lhs(lhs[...])
            loaded_rhs = mask_k_rem_rhs(rhs[...])

            if transpose_rhs:
                dims = (((1,), (1,)), ((), ()))
            else:
                dims = (((1,), (0,)), ((), ()))

            acc_scratch[...] += lax.dot_general(
                loaded_lhs,
                loaded_rhs,
                preferred_element_type=jnp.float32,
                dimension_numbers=dims,
            )
            if is_last_k_tile:
                _store_accum()

        lax.cond(
            k_i == tiles_k - 1,
            functools.partial(_accum, True),
            functools.partial(_accum, False),
        )

    def lhs_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset):
        group_offsets, group_ids, m_tile_ids = group_metadata
        del n_i, group_offsets, group_ids, group_offset
        return m_tile_ids[grid_id], k_i

    def rhs_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset):
        group_offsets, group_ids, m_tile_ids = group_metadata
        del group_offsets, m_tile_ids
        if transpose_rhs:
            k_i, n_i = n_i, k_i
        return group_ids[grid_id] - group_offset[0], k_i, n_i

    def out_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset):
        group_offsets, group_ids, m_tile_ids = group_metadata
        del k_i, group_offsets, group_ids, group_offset
        return m_tile_ids[grid_id], n_i

    out_block_spec = pl.BlockSpec((tm, tn), out_transform_indices)
    if existing_out is None:
        in_out_block_spec: Any = None
        input_output_aliases = {}
    else:
        in_out_block_spec = out_block_spec
        existing_out_arg_index = 6
        input_output_aliases = {existing_out_arg_index: 0}

    lhs_block_spec = pl.BlockSpec((tm, tk), lhs_transform_indices)
    if transpose_rhs:
        rhs_block_spec = pl.BlockSpec((None, tn, tk), rhs_transform_indices)
    else:
        rhs_block_spec = pl.BlockSpec((None, tk, tn), rhs_transform_indices)

    lhs_bytes = _calculate_bytes(lhs)
    rhs_bytes = (k * n) * rhs.itemsize

    out_bytes = (m * n) * jnp.dtype(preferred_element_type).itemsize
    max_active_tiles = group_metadata[1].size
    bytes_accessed = (lhs_bytes * tiles_n) + (rhs_bytes * max_active_tiles) + out_bytes
    flops = 2 * m * k * n
    cost_estimate = pl.CostEstimate(flops=flops, bytes_accessed=bytes_accessed, transcendentals=0)
    metadata = {
        "preferred_element_type": jnp.dtype(preferred_element_type).name,
        "tiling": {"tile_m": tm, "tile_k": tk, "tile_n": tn},
        "transpose_rhs": transpose_rhs,
    }

    call_gmm = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), preferred_element_type),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[
                lhs_block_spec,
                rhs_block_spec,
                in_out_block_spec,
            ],
            out_specs=out_block_spec,
            grid=(tiles_n, num_active_tiles, tiles_k),
            scratch_shapes=[pltpu.VMEM((tm, tn), jnp.float32)],
        ),
        input_output_aliases=input_output_aliases,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary")
        ),
        interpret=interpret,
        cost_estimate=cost_estimate,
        metadata={"xprof_metadata": json.dumps(metadata)},
        # Required so the harness's mosaic runner can capture `module { ... }`.
        debug=True,
    )

    out = call_gmm(
        group_metadata,
        group_offset,
        lhs,
        rhs,
        existing_out,
    )
    if existing_out is None and num_current_groups < num_total_groups:
        out = _zero_uninitialized_memory(
            out,
            start_group=group_offset[0],
            num_nonzero_groups=rhs.shape[0],
            group_metadata=group_metadata,
        )
    return out


def megablox_gmm_kernel(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype = jnp.float32,
    tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128),
    group_offset: jnp.ndarray | None = None,
    existing_out: jnp.ndarray | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
) -> jnp.ndarray:
    """Stable entrypoint name used by the harness registration."""
    return gmm(
        lhs=lhs,
        rhs=rhs,
        group_sizes=group_sizes,
        preferred_element_type=preferred_element_type,
        tiling=tiling,
        group_offset=group_offset,
        existing_out=existing_out,
        transpose_rhs=transpose_rhs,
        interpret=interpret,
    )


def _dtype(name: str) -> jnp.dtype:
    table = {
        "bf16": jnp.bfloat16,
        "f16": jnp.float16,
        "f32": jnp.float32,
    }
    if name not in table:
        raise ValueError(f"Unsupported dtype {name!r}; expected one of {sorted(table)}")
    return table[name]


def _preferred_dtype(name: str) -> jnp.dtype:
    table = {
        "f16": jnp.float16,
        "bf16": jnp.bfloat16,
        "f32": jnp.float32,
    }
    if name not in table:
        raise ValueError(f"Unsupported preferred_element_type {name!r}; expected one of {sorted(table)}")
    return table[name]


def build_args(cfg: Dict[str, Any]) -> Tuple[list, Dict[str, Any]]:
    """Harness adapter contract: returns (args, static_kwargs)."""
    m = int(cfg.get("m", 512))
    k = int(cfg.get("k", 512))
    n = int(cfg.get("n", 512))
    num_groups = int(cfg.get("num_groups", 4))

    tm = int(cfg.get("tm", 128))
    tk = int(cfg.get("tk", 128))
    tn = int(cfg.get("tn", 128))

    dtype = _dtype(str(cfg.get("dtype", "bf16")))
    preferred_element_type = _preferred_dtype(str(cfg.get("preferred_element_type", "f32")))
    transpose_rhs = bool(cfg.get("transpose_rhs", False))
    interpret = bool(cfg.get("interpret", False))

    if m % tm != 0:
        raise ValueError(f"m={m} must be divisible by tm={tm}")

    if "group_sizes" in cfg:
        group_sizes = jnp.asarray(cfg["group_sizes"], dtype=jnp.int32)
    else:
        base = m // num_groups
        rem = m - (base * num_groups)
        sizes = [base] * num_groups
        sizes[-1] += rem
        group_sizes = jnp.asarray(sizes, dtype=jnp.int32)

    if int(group_sizes.sum()) != m:
        raise ValueError("sum(group_sizes) must equal m")

    lhs = jax.ShapeDtypeStruct((m, k), dtype)
    rhs_shape = (num_groups, n, k) if transpose_rhs else (num_groups, k, n)
    rhs = jax.ShapeDtypeStruct(rhs_shape, dtype)

    args = [lhs, rhs, group_sizes]
    static_kwargs = {
        "preferred_element_type": preferred_element_type,
        "tiling": (tm, tk, tn),
        "transpose_rhs": transpose_rhs,
        "interpret": interpret,
    }
    return args, static_kwargs
