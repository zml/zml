const std = @import("std");
const safetensors = @import("../safetensors.zig");

pub const max_read_parallelism: usize = 128;
pub const max_dma_parallelism: usize = 32;
pub const max_read_request_size: usize = 32 * 1024 * 1024;
pub const max_positional_iovecs: usize = safetensors.max_positional_iovecs;

/// The source request size is the larger of the profile's minimum read
/// chunk and the calibrated DMA block, capped at the supported maximum.
pub fn effectiveSourceRequestSize(read_chunk_size: usize, dma_block_size: usize) !usize {
    if (read_chunk_size == 0 or read_chunk_size > max_read_request_size)
        return error.InvalidLoadProfile;
    const selected = @max(read_chunk_size, dma_block_size);
    if (selected > max_read_request_size) return error.InvalidLoadProfile;
    return selected;
}

pub fn maximumCoalescedJobBlocks(request_size: usize, block_size: usize) !usize {
    if (request_size == 0 or block_size == 0) return error.InvalidDmaLoadConfig;
    const scatter_limit = std.math.mul(
        usize,
        block_size,
        max_positional_iovecs,
    ) catch std.math.maxInt(usize);
    const maximum_job_len = @min(request_size, scatter_limit);
    return std.math.divCeil(usize, maximum_job_len, block_size) catch
        error.InvalidDmaLoadConfig;
}
