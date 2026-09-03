const std = @import("std");
const safetensors = @import("../safetensors.zig");

pub const max_read_parallelism: usize = 128;
pub const max_dma_parallelism: usize = 32;
pub const max_read_request_size: usize = 32 * 1024 * 1024;
pub const max_positional_iovecs: usize = safetensors.max_positional_iovecs;

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
