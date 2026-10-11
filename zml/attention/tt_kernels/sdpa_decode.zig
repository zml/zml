//! tt-metal's paged flash-decode attention, ported: the host program factory
//! of `ttnn::transformer::paged_scaled_dot_product_attention_decode`
//! (`ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/
//! sdpa_decode_program_factory.cpp`, tt-metal d04395ed), its device kernels
//! in the TTKernel DSL (`sdpa_decode_kernels.zig`).
//!
//! Supported: causal decode over a cache, paged or not, with positions in a
//! tensor, Q and the output tiled in DRAM. Not ported: MLA, attention masks and sinks,
//! sliding windows, sharded Q/output, half (16x32) tiles; the kernels take the
//! full-tile path they use for more than 16 heads.

const std = @import("std");

const Compiler = @import("../../Compiler.zig");
const zml = @import("../../zml.zig");
const dsl = zml.kernel.ttkernel.dsl;
const program = dsl.program;
const kernels = @import("sdpa_decode_kernels.zig");
const llk = @import("llk.zig");
const Tensor = zml.Tensor;

/// Bytes a DRAM page is aligned to (Blackhole).
const dram_alignment = 64;

/// `MAX_TREE_REDUCTION_ROUNDS` of `sdpa_decode_device_operation.hpp`.
const max_tree_reduction_rounds = 6;
const no_core = std.math.maxInt(u32);

pub const Options = struct {
    scale: f32,
    /// `(cols, rows)` of the cores the program may use, from (0, 0).
    grid: [2]u32 = .{ 8, 8 },
    /// Keys per chunk, a power of two multiple of 32; 0 lets the kernels
    /// choose per position (the paged default).
    k_chunk_size: u32 = 0,
};

/// Causal decode attention: `q` (`[1, B, heads, head_dim]`) over the paged
/// caches `k_cache`, `v_cache` (`[blocks, kv_heads, block_size, head_dim]`),
/// user `b` attending to positions `0..=cur_pos[b]` (`[B]` i32) of its
/// sequence mapped through `page_table` (`[B, max_blocks]` i32).
pub fn pagedSdpaDecode(q: Tensor, k_cache: Tensor, v_cache: Tensor, cur_pos: Tensor, page_table: Tensor, opts: Options) Tensor {
    const allocator = Compiler.current().arena.allocator();
    const art = decodeProgram(allocator, q.shape(), k_cache.shape(), v_cache.shape(), cur_pos.shape(), page_table.shape(), opts) catch |err|
        std.debug.panic("pagedSdpaDecode: {}", .{err});
    return zml.ops.ttKernel(&.{ q, k_cache, v_cache, cur_pos, page_table }, &.{q.shape()}, .{
        .kernel_id = "zml.tt.paged_sdpa_decode",
        .artifact = art,
    })[0];
}

/// Causal decode attention: `q` (`[1, B, heads, head_dim]`) over the caches
/// `k_cache`, `v_cache` (`[B, kv_heads, S, head_dim]`), user `b` attending to
/// positions `0..=cur_pos[b]` (`[B]` i32).
pub fn sdpaDecode(q: Tensor, k_cache: Tensor, v_cache: Tensor, cur_pos: Tensor, opts: Options) Tensor {
    const allocator = Compiler.current().arena.allocator();
    const art = decodeProgram(allocator, q.shape(), k_cache.shape(), v_cache.shape(), cur_pos.shape(), null, opts) catch |err|
        std.debug.panic("sdpaDecode: {}", .{err});
    return zml.ops.ttKernel(&.{ q, k_cache, v_cache, cur_pos }, &.{q.shape()}, .{
        .kernel_id = "zml.tt.sdpa_decode",
        .artifact = art,
    })[0];
}

/// `get_tree_reduction_params` of `sdpa_decode_device_operation.hpp`: core
/// `core_id`'s place in the binary-tree reduction of `n` cores' partials.
const TreeReduction = struct {
    num_rounds: u32 = 0,
    my_active_rounds: u32 = 0,
    is_root: bool = false,
    parent_core_in_group: u32 = no_core,
    send_at_round: u32 = no_core,
    children_per_round: [max_tree_reduction_rounds]u32 = @splat(no_core),
    num_children: u32 = 0,

    fn init(core_id: u32, n: u32) TreeReduction {
        var p: TreeReduction = .{};
        if (n <= 1) {
            p.is_root = true;
            return p;
        }
        const vid = (n - 1) - core_id;
        p.num_rounds = 32 - @clz(n - 1);
        p.is_root = core_id == 0;
        for (0..p.num_rounds) |ri| {
            const r: u5 = @intCast(ri);
            const mask = (@as(u32, 2) << r) - 1;
            if (vid & mask == mask) {
                const child_vid = vid -% (@as(u32, 1) << r);
                if (child_vid < n) {
                    p.children_per_round[ri] = (n - 1) - child_vid;
                    p.num_children += 1;
                    p.my_active_rounds = @intCast(ri + 1);
                }
            }
        }
        if (!p.is_root) {
            const trailing_ones: u5 = @intCast(@ctz(~vid));
            const parent_vid = vid + (@as(u32, 1) << trailing_ones);
            p.parent_core_in_group = if (parent_vid < n) (n - 1) - parent_vid else 0;
            p.send_at_round = trailing_ones;
        }
        if (p.is_root) {
            for (1..n) |c| {
                const cv: u32 = (n - 1) - @as(u32, @intCast(c));
                const t: u5 = @intCast(@ctz(~cv));
                if (cv + (@as(u32, 1) << t) >= n and p.children_per_round[t] == no_core) {
                    p.children_per_round[t] = @intCast(c);
                    p.num_children += 1;
                    p.my_active_rounds = @max(p.my_active_rounds, @as(u32, t) + 1);
                }
            }
        }
        return p;
    }
};

fn log2(x: u32) u32 {
    return 31 - @clz(x);
}

/// The program of `pagedSdpaDecode` (io-tensors `q`, `k_cache`, `v_cache`,
/// `cur_pos`, `page_table`, output) or, without a page table, of
/// `sdpaDecode` (io-tensors `q`, `k_cache`, `v_cache`, `cur_pos`, output).
pub fn decodeProgram(allocator: std.mem.Allocator, q: zml.Shape, k: zml.Shape, v: zml.Shape, cur_pos: zml.Shape, page_table: ?zml.Shape, opts: Options) ![]u8 {
    const io_q = 0;
    const io_k = 1;
    const io_v = 2;
    const io_cur_pos = 3;
    const io_page_table = 4;
    const io_out: u32 = if (page_table != null) 5 else 4;
    const is_paged = page_table != null;

    if (q.dtype() != .bf16 or k.dtype() != .bf16 or v.dtype() != .bf16) return error.UnsupportedDataType;

    // Dimensions.
    const b: u32 = @intCast(if (page_table) |pt| pt.dim(0) else q.dim(1));
    const num_q_heads: u32 = @intCast(q.dim(2));
    const pnh: u32 = std.mem.alignForward(u32, num_q_heads, 32);
    const dh: u32 = @intCast(k.dim(3));
    const vdh: u32 = @intCast(v.dim(3));
    const bkv: u32 = @intCast(k.dim(0));
    const num_kv_heads: u32 = @intCast(k.dim(1));
    var page_block_size_t: u32 = 0;
    var s: u32 = @intCast(k.dim(2));
    if (page_table) |pt| {
        const block_size: u32 = @intCast(k.dim(2));
        if (block_size % 32 != 0) return error.UnsupportedBlockSize;
        page_block_size_t = block_size / 32;
        s = @as(u32, @intCast(pt.dim(-1))) * block_size;
    }

    const st = s / 32;
    const dht = dh / 32;
    const vdht = vdh / 32;
    const pnht = pnh / 32;
    const sk_chunk_t = opts.k_chunk_size / 32;

    // Cores.
    const grid_x = opts.grid[0];
    const grid_y = opts.grid[1];
    const num_cores_available = grid_x * grid_y;
    if (num_cores_available < b) return error.NotEnoughCores;
    const core_grid: program.CoreRange = .grid(grid_x, grid_y);

    const max_cores_per_head = num_cores_available;
    const max_num_cores_for_compute = max_cores_per_head * b * num_kv_heads;
    const num_cores_per_batch_uncapped = @min(num_cores_available, max_num_cores_for_compute) / b;
    const num_cores_per_head = @max(1, num_cores_per_batch_uncapped / num_kv_heads);
    var num_heads_per_core: u32 = @max(1, std.math.divCeil(u32, num_kv_heads, num_cores_per_batch_uncapped) catch unreachable);
    while (num_kv_heads % num_heads_per_core != 0) num_heads_per_core += 1;
    const num_cores_per_batch = num_cores_per_head * num_kv_heads / num_heads_per_core;
    const num_reducer_cores = num_kv_heads * b / num_heads_per_core;
    const num_output_cores = b;
    const num_active_cores = num_cores_per_head * num_kv_heads * b / num_heads_per_core;

    // Compute configuration: HiFi2, approximate math, bf16 DST.
    const fp32_dest_acc_en = false;
    const exp_approx_mode = true;

    const num_tree_reduction_rounds: u32 = if (num_cores_per_head > 1) 32 - @clz(num_cores_per_head - 1) else 0;
    if (num_tree_reduction_rounds > max_tree_reduction_rounds) return error.TooManyCoresPerHead;

    // Linear core assignment (Q in DRAM).
    const core_group = try allocator.alloc(program.CoreCoord, num_cores_available);
    for (core_group, 0..) |*c, i| c.* = .{ .x = @intCast(i % grid_x), .y = @intCast(i / grid_x) };

    const C = program.Arg;
    const v_ = struct {
        fn val(x: u32) C {
            return .{ .value = x };
        }
        fn noc(c: program.CoreCoord, dim: u1) C {
            return .{ .core_noc = .{ .core = c, .dim = dim } };
        }
    };
    const val = v_.val;

    // NoC coordinates of the reducer, output and reduction-group cores.
    const reduce_cores = try allocator.alloc(program.CoreCoord, num_reducer_cores);
    {
        var n: usize = 0;
        for (0..num_active_cores) |i| if (i % num_cores_per_head == 0) {
            reduce_cores[n] = core_group[i];
            n += 1;
        };
    }
    const output_cores = try allocator.alloc(program.CoreCoord, num_output_cores);
    {
        var n: usize = 0;
        for (0..num_active_cores) |i| if (i % num_cores_per_batch == 0) {
            output_cores[n] = core_group[i];
            n += 1;
        };
    }

    // Matmul blocking.
    const dst_size: u32 = if (fp32_dest_acc_en) 4 else 8;
    const max_dynamic_chunk_size = dst_size;
    const sk_chunk_t_cb_size = if (sk_chunk_t == 0) max_dynamic_chunk_size else sk_chunk_t;
    var qk_out_subblock_w: u32 = 0;
    var qk_out_subblock_h: u32 = 0;
    var qk_in0_num_subblocks: u32 = 0;
    var qk_in1_num_subblocks: u32 = 0;
    if (sk_chunk_t > 0) {
        qk_out_subblock_w = @min(sk_chunk_t, dst_size);
        qk_out_subblock_h = if (qk_out_subblock_w == sk_chunk_t) @min(pnht, dst_size / qk_out_subblock_w) else 1;
        qk_in0_num_subblocks = pnht / qk_out_subblock_h;
        qk_in1_num_subblocks = sk_chunk_t / qk_out_subblock_w;
    }
    const out_out_subblock_w = @min(vdht, dst_size);
    const out_out_subblock_h = if (out_out_subblock_w == vdht) @min(pnht, dst_size / out_out_subblock_w) else 1;
    const out_in0_num_subblocks = pnht / out_out_subblock_h;
    const out_in1_num_subblocks = vdht / out_out_subblock_w;
    var dht_granularity = @min(dht, dst_size);
    var log2_dht_granularity = log2(dht_granularity);
    if (dht_granularity != (@as(u32, 1) << @intCast(log2_dht_granularity))) {
        dht_granularity = 1;
        log2_dht_granularity = 0;
    }

    // Circular buffer tiles; every format is bf16 with full tiles here.
    const q_tiles = pnht * dht;
    const k_tiles = sk_chunk_t_cb_size * dht * 2;
    const v_tiles = sk_chunk_t_cb_size * vdht * 2;
    const qk_tiles = pnht * sk_chunk_t_cb_size;
    const out_tiles = pnht * vdht;
    const statistics_tiles = pnht;
    const intermed_output_tiles = (out_tiles + 2 * pnht) * (num_cores_per_head - 1);
    const tile: u32 = program.DataType.bf16.tileSize();

    const cur_pos_stick_size: u32 = std.mem.alignForward(u32, @intCast(cur_pos.byteSize()), dram_alignment);
    const page_table_stick_size: u32 = if (page_table) |pt| std.mem.alignForward(u32, @intCast(pt.dim(-1) * 4), dram_alignment) else 0;

    var cbs: std.ArrayList(program.CircularBuffer) = .empty;
    const cb = struct {
        fn add(list: *std.ArrayList(program.CircularBuffer), a: std.mem.Allocator, index: u32, pages: u32, dtype: program.DataType, page_size: u32, cores: program.CoreRange) !void {
            try list.append(a, .{ .index = index, .dtype = dtype, .page_size = page_size, .num_pages = pages, .cores = cores });
        }
    }.add;
    try cb(&cbs, allocator, 0, q_tiles, .bf16, tile, core_grid); // Q
    try cb(&cbs, allocator, 1, k_tiles, .bf16, tile, core_grid); // K
    try cb(&cbs, allocator, 2, v_tiles, .bf16, tile, core_grid); // V
    try cb(&cbs, allocator, 3, qk_tiles, .bf16, tile, core_grid); // attention mask
    try cb(&cbs, allocator, 5, 1, .bf16, tile, core_grid); // scale
    try cb(&cbs, allocator, 6, statistics_tiles, .bf16, tile, core_grid); // m_in
    try cb(&cbs, allocator, 7, statistics_tiles, .bf16, tile, core_grid); // l_in
    try cb(&cbs, allocator, 8, 1, .si32, cur_pos_stick_size, core_grid); // cur_pos for the writer
    try cb(&cbs, allocator, 15, 1, .si32, cur_pos_stick_size, core_grid); // cur_pos for compute
    if (is_paged) try cb(&cbs, allocator, 9, 1, .si32, page_table_stick_size, core_grid); // page table
    try cb(&cbs, allocator, 10, q_tiles, .bf16, tile, core_grid); // tilized Q
    try cb(&cbs, allocator, 11, 1, .bf16, tile, core_grid); // column identity
    try cb(&cbs, allocator, 12, 1, .bf16, tile, core_grid); // scalar
    try cb(&cbs, allocator, 24, qk_tiles, .bf16, tile, core_grid);
    try cb(&cbs, allocator, 25, out_tiles, .bf16, tile, core_grid);
    try cb(&cbs, allocator, 26, out_tiles, .bf16, tile, core_grid);
    for ([_]u32{ 27, 28, 29, 30, 31, 21, 22 }) |i| try cb(&cbs, allocator, i, statistics_tiles, .bf16, tile, core_grid);
    try cb(&cbs, allocator, 23, out_tiles, .bf16, tile, core_grid);
    try cb(&cbs, allocator, 16, out_tiles, .bf16, tile, core_grid);
    try cb(&cbs, allocator, 17, statistics_tiles, .bf16, tile, core_grid);
    try cb(&cbs, allocator, 18, statistics_tiles, .bf16, tile, core_grid);
    if (intermed_output_tiles > 0) try cb(&cbs, allocator, 19, intermed_output_tiles, .bf16, tile, core_grid);
    try cb(&cbs, allocator, 20, out_tiles, .bf16, tile, core_grid); // output

    // Compile-time args: the kernels' circular buffers, then the accessor
    // args of their tensors; the rest is baked in.
    const ids = kernels.cb;
    var reader_ct: std.ArrayList(C) = .empty;
    for ([_]u32{ ids.q, ids.k, ids.v, ids.writer_cur_pos, ids.compute_cur_pos }) |i| try reader_ct.append(allocator, .{ .cb = i });
    if (is_paged) try reader_ct.append(allocator, .{ .cb = ids.page_table });
    for ([_]u32{ io_q, io_k, io_v, io_cur_pos }) |i| try reader_ct.append(allocator, .{ .tensor_accessor_args = i });
    if (is_paged) try reader_ct.append(allocator, .{ .tensor_accessor_args = io_page_table });

    const reducer_semaphore = 0;
    var writer_ct: std.ArrayList(C) = .empty;
    for ([_]u32{ ids.mask, ids.identity_scale, ids.m_in, ids.l_in, ids.writer_cur_pos, ids.col_identity, ids.zero, ids.out_o, ids.out_m, ids.out_l, ids.intermed, ids.out_final }) |i| try writer_ct.append(allocator, .{ .cb = i });
    try writer_ct.appendSlice(allocator, &.{ val(reducer_semaphore), .{ .tensor_accessor_args = io_out } });

    var compute_ct: std.ArrayList(C) = .empty;
    for (kernels.compute_cbs) |i| try compute_ct.append(allocator, .{ .cb = i });

    const cfg: kernels.Config = .{
        .pnht = pnht,
        .st = st,
        .dht = dht,
        .vdht = vdht,
        .sk_chunk_t = sk_chunk_t,
        .max_dynamic_chunk_size = max_dynamic_chunk_size,
        .num_cores_per_head = num_cores_per_head,
        .num_heads_per_core = num_heads_per_core,
        .num_kv_heads = num_kv_heads,
        .num_q_heads = num_q_heads,
        .paged = is_paged,
        .block_size_t = page_block_size_t,
        .bkv = bkv,
        .tile_bytes = tile,
        .cur_pos_page_bytes = cur_pos_stick_size,
        .page_table_page_bytes = page_table_stick_size,
        .scale_fp32 = @bitCast(opts.scale),
        .identity_scalar_packed = 0x3F80_3F80,
        .num_tree_reduction_rounds = num_tree_reduction_rounds,
        .out_subblock_w = out_out_subblock_w,
        .out_subblock_h = out_out_subblock_h,
        .out_in0_num_subblocks = out_in0_num_subblocks,
        .out_in1_num_subblocks = out_in1_num_subblocks,
        .qk_subblock_w = qk_out_subblock_w,
        .qk_subblock_h = qk_out_subblock_h,
        .qk_in0_num_subblocks = qk_in0_num_subblocks,
        .qk_in1_num_subblocks = qk_in1_num_subblocks,
        .exp_approx = exp_approx_mode,
        .dht_granularity = dht_granularity,
        .sub_exp_granularity = if (sk_chunk_t > 0) @min(sk_chunk_t, dst_size) else null,
    };

    // The defines of `sdpa_flash_decode.cpp` and those `compute_common.hpp`
    // reads.
    var defines: std.ArrayList(program.Define) = .empty;
    try defines.appendSlice(allocator, &.{
        .{ .name = "LLK_ZEROFLAG_OUTLINE", .value = "1" },
        .{ .name = "REDUCE_OP", .value = "(PoolType::MAX)" },
        .{ .name = "REDUCE_DIM", .value = "(ReduceDim::REDUCE_ROW)" },
    });
    try defines.append(allocator, .{ .name = "EXP_APPROX_MODE", .value = if (exp_approx_mode) "1" else "0" });
    try defines.append(allocator, .{ .name = "DHT_GRANULARITY", .value = try std.fmt.allocPrint(allocator, "{d}", .{dht_granularity}) });
    try defines.append(allocator, .{ .name = "LOG2_DHT_GRANULARITY", .value = try std.fmt.allocPrint(allocator, "{d}", .{log2_dht_granularity}) });
    if (sk_chunk_t > 0) {
        for ([_]struct { []const u8, u32 }{
            .{ "SUB_EXP_GRANULARITY", @min(sk_chunk_t, dst_size) },
            .{ "MUL_BCAST_GRANULARITY", @min(pnht * sk_chunk_t, dst_size) },
            .{ "STATS_GRANULARITY", @min(sk_chunk_t, dst_size) },
        }) |g| {
            if (!std.math.isPowerOfTwo(g[1])) return error.InvalidGranularity;
            try defines.append(allocator, .{ .name = g[0], .value = try std.fmt.allocPrint(allocator, "{d}", .{g[1]}) });
            try defines.append(allocator, .{ .name = try std.fmt.allocPrint(allocator, "LOG2_{s}", .{g[0]}), .value = try std.fmt.allocPrint(allocator, "{d}", .{log2(g[1])}) });
        }
    } else {
        try defines.append(allocator, .{ .name = "DYNAMIC_CHUNK_SIZE", .value = "1" });
    }

    // Runtime arguments.
    var reader_rt: std.ArrayList(program.CoreArgs) = .empty;
    var writer_rt: std.ArrayList(program.CoreArgs) = .empty;
    var compute_rt: std.ArrayList(program.CoreArgs) = .empty;
    for (core_group, 0..) |core, ii| {
        const i: u32 = @intCast(ii);
        if (i >= num_active_cores) {
            // Idle core.
            const r = try allocator.alloc(C, 20);
            @memset(r, val(0));
            const w = try allocator.alloc(C, 10 + 6 + max_tree_reduction_rounds + 2 * num_cores_per_head);
            @memset(w, val(0));
            const c = try allocator.alloc(C, 7 + 5 + max_tree_reduction_rounds);
            @memset(c, val(0));
            c[0] = val(65);
            try reader_rt.append(allocator, .{ .core = core, .args = r });
            try writer_rt.append(allocator, .{ .core = core, .args = w });
            try compute_rt.append(allocator, .{ .core = core, .args = c });
            continue;
        }
        const cur_head = (i % num_cores_per_batch) / num_cores_per_head;
        const cur_batch = i / num_cores_per_batch;
        const core_num_in_reduce = i % num_cores_per_head;
        const core_num_in_output = i % num_cores_per_batch;
        const worker_id_for_reduce = core_num_in_reduce -% 1;
        const worker_id_for_output = if (core_num_in_output == 0) no_core else core_num_in_output - 1;
        const do_reduce: u32 = @intFromBool(worker_id_for_reduce == no_core);
        const do_output: u32 = @intFromBool(worker_id_for_output == no_core);
        const cur_pos_arg = no_core; // read from the cur_pos tensor
        const tree: TreeReduction = .init(core_num_in_reduce, num_cores_per_head);
        const reduction_group_base_idx = cur_batch * num_cores_per_batch + cur_head * num_cores_per_head;

        var r: std.ArrayList(C) = .empty;
        try r.appendSlice(allocator, &.{
            .{ .tensor_address = io_q },       .{ .tensor_address = io_k },                                    .{ .tensor_address = io_v },
            .{ .tensor_address = io_cur_pos }, if (is_paged) .{ .tensor_address = io_page_table } else val(0), val(0),
            val(0),                            val(page_table_stick_size),                                     val(do_reduce),
            val(do_output),                    val(cur_head),                                                  val(cur_batch),
            val(core_num_in_reduce),           val(core_num_in_output),                                        val(cur_pos_arg),
            val(0),                            val(0),                                                         val(0),
            val(0),                            val(0),
        });
        for (output_cores) |oc| try r.append(allocator, v_.noc(oc, 0));
        for (output_cores) |oc| try r.append(allocator, v_.noc(oc, 1));

        var w: std.ArrayList(C) = .empty;
        try w.appendSlice(allocator, &.{
            .{ .tensor_address = io_out }, val(worker_id_for_reduce), val(worker_id_for_output),       val(do_reduce),
            val(do_output),                val(cur_head),             val(cur_batch),                  val(core_num_in_reduce),
            val(core_num_in_output),       val(cur_pos_arg),          val(@intFromBool(tree.is_root)), val(tree.parent_core_in_group),
            val(tree.send_at_round),       val(tree.num_children),    val(tree.my_active_rounds),      val(reduction_group_base_idx),
        });
        for (tree.children_per_round) |child| try w.append(allocator, val(child));
        for (0..num_cores_per_head) |c| try w.append(allocator, v_.noc(core_group[reduction_group_base_idx + c], 0));
        for (0..num_cores_per_head) |c| try w.append(allocator, v_.noc(core_group[reduction_group_base_idx + c], 1));
        for (reduce_cores) |rc| try w.append(allocator, v_.noc(rc, 0));
        for (reduce_cores) |rc| try w.append(allocator, v_.noc(rc, 1));
        for (output_cores) |oc| try w.append(allocator, v_.noc(oc, 0));
        for (output_cores) |oc| try w.append(allocator, v_.noc(oc, 1));

        var c: std.ArrayList(C) = .empty;
        try c.appendSlice(allocator, &.{
            val(do_reduce),                 val(do_output),          val(cur_head),          val(cur_batch),
            val(core_num_in_reduce),        val(core_num_in_output), val(cur_pos_arg),       val(@intFromBool(tree.is_root)),
            val(tree.parent_core_in_group), val(tree.send_at_round), val(tree.num_children), val(tree.my_active_rounds),
        });
        for (tree.children_per_round) |child| try c.append(allocator, val(child));

        try reader_rt.append(allocator, .{ .core = core, .args = r.items });
        try writer_rt.append(allocator, .{ .core = core, .args = w.items });
        try compute_rt.append(allocator, .{ .core = core, .args = c.items });
    }

    const ctx = try dsl.newContext();
    defer ctx.deinit();
    const p: program.Program = .{
        .kernels = &.{
            .{
                .thread = .reader,
                .source = try llk.source(allocator, ctx, "sdpa_decode_reader", .reader, kernels.reader, cfg),
                .cores = core_grid,
                .compile_time_args = reader_ct.items,
                .runtime_args = reader_rt.items,
            },
            .{
                .thread = .writer,
                .source = try llk.source(allocator, ctx, "sdpa_decode_writer", .writer, kernels.writer, cfg),
                .cores = core_grid,
                .compile_time_args = writer_ct.items,
                .runtime_args = writer_rt.items,
            },
            .{
                .thread = .{ .compute = .{ .math_fidelity = .hifi2, .fp32_dest_acc_en = fp32_dest_acc_en, .math_approx_mode = true } },
                .source = try llk.source(allocator, ctx, "sdpa_flash_decode", .compute, kernels.compute, cfg),
                .defines = defines.items,
                .cores = core_grid,
                .compile_time_args = compute_ct.items,
                .runtime_args = compute_rt.items,
            },
        },
        .circular_buffers = cbs.items,
        .semaphores = &.{.{ .cores = core_grid }},
        .operands = if (is_paged) &.{
            .{ .index = io_cur_pos, .layout = .row_major },
            .{ .index = io_page_table, .layout = .row_major },
        } else &.{
            .{ .index = io_cur_pos, .layout = .row_major },
        },
    };
    return p.artifact(allocator);
}

test "tree reduction matches tt-metal" {
    // 4 cores: 0 is the root receiving 1 (round 1) and 2 (round 0)... per
    // vid = 3 - core: vid 3 (core 0) receives vid 2 (core 1) at round 0 and
    // vid 1 (core 2) at round 1; vid 0 (core 3) sends to vid 1 (core 2).
    const root: TreeReduction = .init(0, 4);
    try std.testing.expect(root.is_root);
    try std.testing.expectEqual(1, root.children_per_round[0]);
    try std.testing.expectEqual(2, root.children_per_round[1]);
    const leaf: TreeReduction = .init(3, 4);
    try std.testing.expectEqual(2, leaf.parent_core_in_group);
    try std.testing.expectEqual(0, leaf.send_at_round);
}
