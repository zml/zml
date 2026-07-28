const std = @import("std");
const builtin = @import("builtin");

pub const warp_size: u32 = 32;
pub const default_warps_per_block: u32 = 8;
pub const default_threads_per_block: u32 = warp_size * default_warps_per_block;

const is_nvptx = builtin.cpu.arch.isNvptx();

pub const SharedMem = opaque {};
pub extern var shared_memory: SharedMem align(64) addrspace(.shared);

pub const GridBarrier = extern struct {
    arrivals: u32 = 0,
    phase: u32 = 0,
};

/// Expose all dynamic shared memory assigned to the persistent kernel.
pub fn sharedMemory(comptime T: type) []align(64) addrspace(.shared) T {
    const mem_u8: [*]align(64) addrspace(.shared) u8 = @ptrCast(&shared_memory);
    return @ptrCast(mem_u8[0..totalSharedMemory()]);
}

pub fn totalSharedMemory() u32 {
    if (comptime !is_nvptx) return 0;
    return asm ("mov.u32 %[result], %total_smem_size;"
        : [result] "=r" (-> u32),
    );
}

/// The logical worker coordinates used by every composable sub-kernel.
pub const Grid = struct {
    block_id: u32,
    block_count: u32,
    thread_id: u32,
    threads_per_block: u32,

    pub fn init() Grid {
        if (comptime !is_nvptx) {
            return .{
                .block_id = 0,
                .block_count = 1,
                .thread_id = 0,
                .threads_per_block = 1,
            };
        }
        return .{
            .block_id = specialRegister("%ctaid.x"),
            .block_count = specialRegister("%nctaid.x"),
            .thread_id = specialRegister("%tid.x"),
            .threads_per_block = specialRegister("%ntid.x"),
        };
    }

    pub fn globalThread(self: Grid) u32 {
        return self.block_id * self.threads_per_block + self.thread_id;
    }

    pub fn threadCount(self: Grid) u32 {
        return self.block_count * self.threads_per_block;
    }

    pub fn lane(self: Grid) u32 {
        return self.thread_id % warp_size;
    }

    pub fn globalWarp(self: Grid) u32 {
        return self.block_id * (self.threads_per_block / warp_size) +
            self.thread_id / warp_size;
    }

    pub fn warpCount(self: Grid) u32 {
        return self.block_count * (self.threads_per_block / warp_size);
    }
};

fn specialRegister(comptime register: []const u8) u32 {
    return asm ("mov.u32 %[result], " ++ register ++ ";"
        : [result] "=r" (-> u32),
    );
}

fn syncThreads() void {
    if (comptime is_nvptx) {
        asm volatile ("bar.sync 0;" ::: .{ .memory = true });
    }
}

/// Synchronize every persistent block between graph operations.
///
/// The launcher admits exactly one block per SM, so every block participating
/// in this software barrier is resident and can make forward progress.
pub fn gridBarrier(barrier: *GridBarrier) void {
    if (comptime !is_nvptx) return;

    const grid: Grid = .init();
    syncThreads();
    if (grid.thread_id == 0) {
        const phase = @atomicLoad(u32, &barrier.phase, .monotonic);
        asm volatile ("membar.gl;" ::: .{ .memory = true });
        const previous = @atomicRmw(u32, &barrier.arrivals, .Add, 1, .monotonic);
        if (previous + 1 == grid.block_count) {
            @atomicStore(u32, &barrier.arrivals, 0, .monotonic);
            asm volatile ("membar.gl;" ::: .{ .memory = true });
            @atomicStore(u32, &barrier.phase, phase ^ 1, .monotonic);
        } else {
            while (@atomicLoad(u32, &barrier.phase, .monotonic) == phase) {
                asm volatile ("" ::: .{ .memory = true });
            }
        }
        asm volatile ("membar.gl;" ::: .{ .memory = true });
    }
    syncThreads();
}

fn shuffleDown(value: f32, offset: u32) f32 {
    if (comptime !is_nvptx) return value;
    return asm ("shfl.sync.down.b32 %[result], %[value], %[offset], 0x1f, 0xffffffff;"
        : [result] "=f" (-> f32),
        : [value] "f" (value),
          [offset] "r" (offset),
    );
}

fn warpSum(value_: f32) f32 {
    var value = value_;
    inline for (.{ 16, 8, 4, 2, 1 }) |offset| {
        value += shuffleDown(value, offset);
    }
    return value;
}

/// Compute a row-major `m x k` by `k x n` f32 matrix product.
///
/// One warp owns one output element. The fixed persistent grid strides over
/// output elements, while lanes split the contracting dimension.
pub fn matmulF32(
    comptime Rhs: type,
    lhs: [*]const f32,
    rhs: [*]const Rhs,
    output: [*]f32,
    m: usize,
    n: usize,
    k: usize,
) void {
    if (comptime !is_nvptx) {
        for (0..m) |row| {
            for (0..n) |column| {
                var sum: f32 = 0;
                for (0..k) |contracting| {
                    sum += lhs[row * k + contracting] *
                        asF32(Rhs, rhs[contracting * n + column]);
                }
                output[row * n + column] = sum;
            }
        }
        return;
    }

    const grid: Grid = .init();
    var output_index: usize = grid.globalWarp();
    const output_stride: usize = grid.warpCount();
    while (output_index < m * n) : (output_index += output_stride) {
        const row = output_index / n;
        const column = output_index % n;
        var sum: f32 = 0;
        var contracting: usize = grid.lane();
        while (contracting < k) : (contracting += warp_size) {
            sum += lhs[row * k + contracting] *
                asF32(Rhs, rhs[contracting * n + column]);
        }
        sum = warpSum(sum);
        if (grid.lane() == 0) output[output_index] = sum;
    }
}

fn asF32(comptime T: type, value: T) f32 {
    return switch (@typeInfo(T)) {
        .int => @floatFromInt(value),
        .float => @floatCast(value),
        else => @compileError("matmulF32 only supports integer and float RHS values"),
    };
}

/// Add a row-major bias vector along the last matrix dimension in place.
pub fn addBiasF32(values: [*]f32, bias: [*]const f32, rows: usize, columns: usize) void {
    const grid: Grid = .init();
    var index: usize = grid.globalThread();
    const stride: usize = grid.threadCount();
    while (index < rows * columns) : (index += stride) {
        values[index] += bias[index % columns];
    }
}

/// Fuse bias addition and ReLU into one grid-stride pass over the matrix.
pub fn addBiasReluF32(
    values: [*]f32,
    bias: [*]const f32,
    rows: usize,
    columns: usize,
) void {
    const grid: Grid = .init();
    var index: usize = grid.globalThread();
    const stride: usize = grid.threadCount();
    while (index < rows * columns) : (index += stride) {
        values[index] = @max(values[index] + bias[index % columns], 0);
    }
}

/// Apply ReLU to a contiguous f32 tensor in place.
pub fn reluF32(values: [*]f32, len: usize) void {
    const grid: Grid = .init();
    var index: usize = grid.globalThread();
    const stride: usize = grid.threadCount();
    while (index < len) : (index += stride) {
        values[index] = @max(values[index], 0);
    }
}

/// Return the first index of the greatest value.
///
/// Argmax is the terminal reduction in the current mega-kernel, so block zero
/// consumes the fixed grid's shared-memory arena while the other blocks retire.
pub fn argMaxF32(
    comptime Index: type,
    values: [*]const f32,
    len: usize,
    output: *Index,
) void {
    if (comptime !is_nvptx) {
        var best_index: u32 = 0;
        var best_value = -std.math.inf(f32);
        for (values[0..len], 0..) |value, index| {
            if (value > best_value) {
                best_value = value;
                best_index = @intCast(index);
            }
        }
        output.* = @intCast(best_index);
        return;
    }

    const grid: Grid = .init();
    if (grid.block_id != 0) return;

    var best_index: u32 = 0;
    var best_value = -std.math.inf(f32);
    var index: usize = grid.thread_id;
    while (index < len) : (index += grid.threads_per_block) {
        const value = values[index];
        if (value > best_value) {
            best_value = value;
            best_index = @intCast(index);
        }
    }

    const shared = sharedMemory(u8);
    const values_bytes = grid.threads_per_block * @sizeOf(f32);
    const partial_values: [*]addrspace(.shared) f32 = @ptrCast(shared.ptr);
    const partial_indices: [*]addrspace(.shared) u32 = @ptrCast(@alignCast(
        shared.ptr + values_bytes,
    ));
    partial_values[grid.thread_id] = best_value;
    partial_indices[grid.thread_id] = best_index;
    syncThreads();

    var active = grid.threads_per_block;
    while (active > 1) {
        const next = (active + 1) / 2;
        if (grid.thread_id < active / 2) {
            const other = grid.thread_id + next;
            const other_value = partial_values[other];
            const current_value = partial_values[grid.thread_id];
            if (other_value > current_value or
                (other_value == current_value and
                    partial_indices[other] < partial_indices[grid.thread_id]))
            {
                partial_values[grid.thread_id] = other_value;
                partial_indices[grid.thread_id] = partial_indices[other];
            }
        }
        syncThreads();
        active = next;
    }
    if (grid.thread_id == 0) output.* = @intCast(partial_indices[0]);
}

test "compose dense layer and argmax sub-kernels on host" {
    const lhs = [_]f32{
        1, 2, 3,
        3, 2, 1,
    };
    const rhs = [_]f32{ 1, -1, 2 };
    var result: [2]f32 = undefined;
    matmulF32(f32, &lhs, &rhs, &result, 2, 1, 3);
    addBiasReluF32(&result, &[_]f32{1}, 2, 1);
    try std.testing.expectEqualSlices(f32, &.{ 6, 4 }, &result);

    var index: u32 = undefined;
    argMaxF32(u32, &result, result.len, &index);
    try std.testing.expectEqual(@as(u32, 0), index);
}
