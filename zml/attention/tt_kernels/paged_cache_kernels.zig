//! The device kernels of tt-metal's paged cache operations
//! (`experimental/paged_cache/device/kernels/`, tt-metal d04395ed) in the
//! TTKernel DSL, for the configurations the port runs: the update with an
//! index tensor over a paged cache, without the shared-cache signal; the
//! fill with a batch index tensor, without a valid length or a capacity.
//!
//! A kernel's circular buffers are its first compile-time arguments, then
//! the accessor arguments of its tensors; sizes are built in.

const std = @import("std");

const zml = @import("../../zml.zig");
const dsl = zml.kernel.ttkernel.dsl;
const llk = @import("llk.zig");
const Builder = dsl.Builder;
const Value = dsl.Value;

/// Circular buffer indices of the update program.
pub const update_cb = struct {
    pub const cache = 0;
    pub const input = 1;
    pub const index = 2;
    pub const page_table = 3;
    pub const out = 16;
    /// The untilized cache rows, written by the writer as the next one.
    pub const interm0 = 24;
    pub const interm1 = 25;
    /// The untilized input rows.
    pub const interm2 = 26;
};

/// Circular buffer indices of the fill program.
pub const fill_cb = struct {
    pub const input = 0;
    pub const page_table = 1;
    pub const batch_idx = 2;
};

pub const UpdateConfig = struct {
    /// Tiles of a head's row, heads, cache block rows, in tiles.
    wt: u32,
    num_heads: u32,
    block_size: u32,
    block_size_t: u32,
    /// Bytes of a row of the input.
    wbytes: u32,
    index_page_bytes: u32,
    page_table_page_bytes: u32,
};

fn runtimeArgs(comptime n: usize, k: *Builder, first: u32) [n]Value {
    var out: [n]Value = undefined;
    for (&out, 0..) |*v, i| v.* = k.runtimeArg(first + @as(u32, @intCast(i)));
    return out;
}

/// The first cache tile of a user's update position `idx`: its row of
/// tiles in its block, through the page table at L1 `page_table`.
fn updateCacheTile(k: *Builder, c: UpdateConfig, idx: Value, page_table: Value) Value {
    const block = k.loadL1(k.l1Ptr(page_table, 32), idx.div(c.block_size), 32);
    const row = idx.rem(c.block_size).div(32);
    return block.mul(c.num_heads * c.block_size_t * c.wt).add(row.mul(c.wt));
}

/// `reader_update_cache_interleaved_start_id.cpp`: compile-time args the
/// cache, input, index and page table buffers, then the accessor args of
/// the cache, the index and the page table. Runtime args: the cache's
/// address, its first tile, the index's address, this core's user, the page
/// table's address.
pub fn updateReader(k: *Builder, c: UpdateConfig) void {
    const cb_cache = k.circularBufferOf(0, .bf16, 1);
    const cb_input = k.circularBufferOf(1, .bf16, 1);
    const cb_index = k.circularBufferOf(2, .si32, 1);
    const cb_page_table = k.circularBufferOf(3, .si32, 1);
    const cache_addr, const cache_start, const index_addr, const user, const page_table_addr = runtimeArgs(5, k, 0);
    const noc = k.kernelNoc();

    // The input is this core's shard: its pages are there.
    k.op(.cb_reserve_back, .{ cb_input, c.wt });
    k.op(.cb_push_back, .{ cb_input, c.wt });
    const tile_bytes = k.op(.get_tile_size, .{cb_cache});
    const cache_args = k.tensorAccessorArgs(4, 0);
    const index_args = k.nextTensorAccessorArgs(cache_args);
    const s0 = k.tensorAccessor(cache_args, cache_addr, tile_bytes);
    const index_acc = k.tensorAccessor(index_args, index_addr, c.index_page_bytes);
    const page_table_acc = k.tensorAccessor(k.nextTensorAccessorArgs(index_args), page_table_addr, c.page_table_page_bytes);

    k.op(.cb_reserve_back, .{ cb_index, 1 });
    const index_l1 = k.op(.get_write_ptr, .{cb_index});
    k.op(.noc_async_read_tile, .{ 0, index_acc, index_l1, noc });
    k.op(.noc_async_read_barrier, .{noc});
    k.op(.cb_push_back, .{ cb_index, 1 });
    const idx = k.loadL1(k.l1Ptr(index_l1, 32), user, 32);
    // Position -1 leaves the user's cache alone.
    const update = idx.ne(-1);
    const cache_id = k.variable(cache_start);
    var if_update = k.openIf(update);
    {
        k.op(.cb_reserve_back, .{ cb_page_table, 1 });
        const page_table_l1 = k.op(.get_write_ptr, .{cb_page_table});
        k.op(.noc_async_read_tile, .{ user, page_table_acc, page_table_l1, noc });
        k.op(.noc_async_read_barrier, .{noc});
        k.op(.cb_push_back, .{ cb_page_table, 1 });
        cache_id.set(updateCacheTile(k, c, idx, page_table_l1));
    }
    if_update.yieldThen(.{});

    var heads, const head = llk.range(k, c.num_heads);
    {
        k.op(.cb_reserve_back, .{ cb_cache, c.wt });
        var if_read = k.openIf(update);
        {
            const l1 = k.op(.get_write_ptr, .{cb_cache});
            const first = cache_id.get().add(head.mul(c.wt * c.block_size_t));
            var tiles, const t = llk.range(k, c.wt);
            k.op(.noc_async_read_tile, .{ first.add(t), s0, l1.add(t.mul(tile_bytes)), noc });
            tiles.yield(.{});
            k.op(.noc_async_read_barrier, .{noc});
        }
        if_read.yieldThen(.{});
        k.op(.cb_push_back, .{ cb_cache, c.wt });
    }
    heads.yield(.{});
}

/// `writer_update_cache_interleaved_start_id.cpp`: compile-time args the
/// output, the three untilized buffers, the index and page table buffers,
/// then the accessor args of the cache. Runtime args: the cache's address,
/// its first tile, the row's byte offset in its tile, this core's user.
pub fn updateWriter(k: *Builder, c: UpdateConfig) void {
    const cb_out = k.circularBufferOf(0, .bf16, 1);
    const cb_untilized_cache = k.circularBufferOf(1, .bf16, 1);
    const cb_untilized_cache2 = k.circularBufferOf(2, .bf16, 1);
    const cb_untilized_input = k.circularBufferOf(3, .bf16, 1);
    const cb_index = k.circularBufferOf(4, .si32, 1);
    const cb_page_table = k.circularBufferOf(5, .si32, 1);
    const cache_addr, const cache_start, const row_offset, const user = runtimeArgs(4, k, 0);
    const noc = k.kernelNoc();

    const tile_bytes = k.op(.get_tile_size, .{cb_out});
    const s0 = k.tensorAccessor(k.tensorAccessorArgs(6, 0), cache_addr, tile_bytes);
    k.op(.cb_wait_front, .{ cb_index, 1 });
    const idx = k.loadL1(k.l1Ptr(k.op(.get_read_ptr, .{cb_index}), 32), user, 32);
    const update = idx.ne(-1);
    const cache_id = k.variable(cache_start);
    const offset = k.variable(row_offset);
    var if_update = k.openIf(update);
    {
        k.op(.cb_wait_front, .{ cb_page_table, 1 });
        cache_id.set(updateCacheTile(k, c, idx, k.op(.get_read_ptr, .{cb_page_table})));
        k.op(.cb_pop_front, .{ cb_page_table, 1 });
        offset.set(idx.rem(32).mul(c.wbytes));
    }
    if_update.yieldThen(.{});
    k.op(.cb_pop_front, .{ cb_index, 1 });

    k.op(.cb_wait_front, .{ cb_untilized_input, c.wt });
    const input_l1 = k.op(.get_read_ptr, .{cb_untilized_input});
    var heads, const head = llk.range(k, c.num_heads);
    {
        // The input row into the untilized cache rows, through the NoC.
        k.op(.cb_wait_front, .{ cb_untilized_cache, c.wt });
        k.op(.cb_reserve_back, .{ cb_untilized_cache2, c.wt });
        const dst = k.op(.get_read_ptr, .{cb_untilized_cache}).add(offset.get());
        llk.localCopy(k, input_l1.add(head.mul(c.wbytes)), dst, c.wbytes);
        k.op(.cb_push_back, .{ cb_untilized_cache2, c.wt });
        k.op(.cb_pop_front, .{ cb_untilized_cache, c.wt });

        k.op(.cb_wait_front, .{ cb_out, c.wt });
        var if_write = k.openIf(update);
        {
            const l1 = k.op(.get_read_ptr, .{cb_out});
            const first = cache_id.get().add(head.mul(c.wt * c.block_size_t));
            var tiles, const t = llk.range(k, c.wt);
            k.op(.noc_async_write_tile, .{ first.add(t), s0, l1.add(t.mul(tile_bytes)), noc });
            tiles.yield(.{});
            k.op(.noc_async_writes_flushed, .{noc});
        }
        if_write.yieldThen(.{});
        k.op(.cb_pop_front, .{ cb_out, c.wt });
        var if_barrier = k.openIf(update);
        k.op(.noc_async_write_barrier, .{noc});
        if_barrier.yieldThen(.{});
    }
    heads.yield(.{});
    k.op(.cb_pop_front, .{ cb_untilized_input, c.wt });
}

const untilize_h = "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp";
const tilize_h = "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp";

/// `update_cache.cpp`: untilizes the input once, then per head the cache
/// rows, and tilizes them back with the input row in. Compile-time args the
/// input and untilized input buffers.
pub fn updateCompute(k: *Builder, c: UpdateConfig) void {
    const cb_input = k.circularBufferOf(0, .bf16, 1);
    const cb_untilized_input = k.circularBufferOf(1, .bf16, 1);
    const u = update_cb;

    k.op(.compute_kernel_hw_startup, .{ cb_input, null, cb_untilized_input });
    k.call(untilize_h, "compute_kernel_lib::untilize", .{
        c.wt,
        u.input,
        u.interm2,
        "compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit",
        "compute_kernel_lib::untilize_config::WaitMode::WaitBlock",
        "compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure",
    }, .{1});
    var heads, _ = llk.range(k, c.num_heads);
    k.call(untilize_h, "compute_kernel_lib::untilize", .{ c.wt, u.cache, u.interm0 }, .{1});
    k.call(tilize_h, "compute_kernel_lib::tilize", .{ c.wt, u.interm1, u.out }, .{1});
    heads.yield(.{});
}

pub const FillConfig = struct {
    wt: u32,
    num_heads: u32,
    block_size_t: u32,
    /// Tile rows of a head, of a user.
    num_blocks_of_work_per_head: u32,
    num_blocks_per_batch: u32,
    /// Users; more than one: a batch index each.
    batch_idx_num_elements: u32,
    batch_idx_page_bytes: u32,
    page_table_page_bytes: u32,
};

/// `reader_fill_cache_interleaved.cpp`: compile-time args the input buffer,
/// then the input's accessor args. Runtime args: the input's address, its
/// first tile, rows of tiles.
pub fn fillReader(k: *Builder, c: FillConfig) void {
    const cb_in = k.circularBufferOf(0, .bf16, 1);
    const src_addr, const start_tile, const num_rows = runtimeArgs(3, k, 0);
    const tile_bytes = k.op(.get_tile_size, .{cb_in});
    const s = k.tensorAccessor(k.tensorAccessorArgs(1, 0), src_addr, tile_bytes);
    var rows, const row = llk.range(k, num_rows);
    llk.readTiles(k, s, cb_in, start_tile.add(row.mul(c.wt)), c.wt, tile_bytes);
    rows.yield(.{});
}

/// `writer_fill_cache_interleaved.cpp`: compile-time args the input, page
/// table and batch index buffers, then the accessor args of the cache, the
/// page table and the batch index. Runtime args: the cache's and page
/// table's addresses, the first row, rows, the batch index's address.
pub fn fillWriter(k: *Builder, c: FillConfig) void {
    const cb_in = k.circularBufferOf(0, .bf16, 1);
    const cb_page_table = k.circularBufferOf(1, .si32, 1);
    const cb_batch_idx = k.circularBufferOf(2, .si32, 1);
    const dst_addr, const page_table_addr, const start_row, const num_rows, const batch_idx_addr = runtimeArgs(5, k, 0);
    const noc = k.kernelNoc();
    const batched = c.batch_idx_num_elements > 1;

    const tile_bytes = k.op(.get_tile_size, .{cb_in});
    const out_args = k.tensorAccessorArgs(3, 0);
    const page_table_args = k.nextTensorAccessorArgs(out_args);
    const out = k.tensorAccessor(out_args, dst_addr, tile_bytes);
    const page_table_acc = k.tensorAccessor(page_table_args, page_table_addr, c.page_table_page_bytes);
    const batch_idx_acc = k.tensorAccessor(k.nextTensorAccessorArgs(page_table_args), batch_idx_addr, c.batch_idx_page_bytes);

    k.op(.cb_reserve_back, .{ cb_batch_idx, 1 });
    const batch_idx_l1 = k.op(.get_write_ptr, .{cb_batch_idx});
    k.op(.noc_async_read_tile, .{ 0, batch_idx_acc, batch_idx_l1, noc });
    k.op(.noc_async_read_barrier, .{noc});
    const batch_idxs = k.l1Ptr(batch_idx_l1, 32);
    k.op(.cb_reserve_back, .{ cb_page_table, 1 });
    const page_table_l1 = k.op(.get_write_ptr, .{cb_page_table});
    const page_table = k.l1Ptr(page_table_l1, 32);
    const cached_batch = k.variable(-1);

    var rows, const r = llk.range(k, num_rows);
    {
        const row = start_row.add(r);
        const cur_batch = if (batched) row.div(c.num_blocks_per_batch) else k.int(0);
        const row_in_batch = row.sub(cur_batch.mul(c.num_blocks_per_batch));
        const head = row_in_batch.div(c.num_blocks_of_work_per_head);
        const seq_tile = row_in_batch.rem(c.num_blocks_of_work_per_head);
        const batch_idx = k.loadL1(batch_idxs, cur_batch, 32);
        // A user's page table row, read when the user changes.
        var if_new = k.openIf(batch_idx.ne(cached_batch.get()));
        {
            k.op(.noc_async_read_tile, .{ batch_idx, page_table_acc, page_table_l1, noc });
            k.op(.noc_async_read_barrier, .{noc});
            cached_batch.set(batch_idx);
        }
        if_new.yieldThen(.{});
        const block = k.loadL1(page_table, seq_tile.div(c.block_size_t), 32);
        k.op(.cb_wait_front, .{ cb_in, c.wt });
        // Block -1: not cached.
        var if_cached = k.openIf(block.ne(-1));
        {
            const first = block.mul(c.num_heads * c.block_size_t * c.wt)
                .add(head.mul(c.block_size_t * c.wt))
                .add(seq_tile.rem(c.block_size_t).mul(c.wt));
            const l1 = k.op(.get_read_ptr, .{cb_in});
            var tiles, const t = llk.range(k, c.wt);
            k.op(.noc_async_write_tile, .{ first.add(t), out, l1.add(t.mul(tile_bytes)), noc });
            tiles.yield(.{});
            k.op(.noc_async_write_barrier, .{noc});
        }
        if_cached.yieldThen(.{});
        k.op(.cb_pop_front, .{ cb_in, c.wt });
    }
    rows.yield(.{});
}
