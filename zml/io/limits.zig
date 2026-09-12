const std = @import("std");
const safetensors = @import("../safetensors.zig");

pub const max_read_parallelism: usize = 128;
pub const max_dma_parallelism: usize = 32;

/// Concurrent source reads by latency class, from the recorded width sweeps
/// (CTX.md, fifteenth pass). Local sources knee at 12 to 16 on every host
/// measured: the B70 peaks at 12 to 16 and loses 11% at 24, one MI300X
/// prefers 12, the RTX 5090 host 16 over 12, and the GB300 plateau from 12
/// to 48 has 16 within 5% of its best rung, so 16 is the width with the
/// smallest regret. Remote sources plateau from 24 to 32: hf:// reads 32
/// and 64 alike and AWS was flat from 24, while pinned memory grows with
/// the width (32 x 32 MiB requests pin 1.5 GiB).
pub const local_read_parallelism: usize = 16;
pub const high_latency_read_parallelism: usize = 32;

pub fn defaultReadParallelism(high_latency: bool) usize {
    return if (high_latency) high_latency_read_parallelism else local_read_parallelism;
}
pub const max_read_request_size: usize = 32 * 1024 * 1024;
pub const max_positional_iovecs: usize = safetensors.max_positional_iovecs;

/// The source request size is the larger of the profile's minimum read
/// chunk and the calibrated DMA block, capped at the supported maximum.
pub fn effectiveSourceRequestSize(read_chunk_size: usize, dma_block_size: usize) error{InvalidOptions}!usize {
    if (read_chunk_size == 0 or read_chunk_size > max_read_request_size)
        return error.InvalidOptions;
    const selected = @max(read_chunk_size, dma_block_size);
    if (selected > max_read_request_size) return error.InvalidOptions;
    return selected;
}

pub fn maximumCoalescedJobBlocks(request_size: usize, block_size: usize) error{InvalidOptions}!usize {
    if (request_size == 0 or block_size == 0) return error.InvalidOptions;
    const scatter_limit = block_size *| max_positional_iovecs;
    const maximum_job_len = @min(request_size, scatter_limit);
    return maximum_job_len / block_size + @intFromBool(maximum_job_len % block_size != 0);
}
