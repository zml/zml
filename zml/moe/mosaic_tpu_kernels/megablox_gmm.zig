const std = @import("std");

const zml = @import("../../zml.zig");

const mtt = @import("kernels/mosaic_tpu/builder");
const mtt_kernel = zml.kernel.mosaic_tpu;

pub const Cfg = struct {
    m: usize,
    k: usize,
    n: usize,
    num_groups: usize,
    num_active_tiles: usize,

    tm: usize = 128,
    tk: usize = 128,
    tn: usize = 128,

    dtype: mtt.DType,
    preferred_element_type: mtt.DType,
    transpose_rhs: bool = true,
};

fn rhsWindowShape(cfg: Cfg) [2]i64 {
    return if (cfg.transpose_rhs)
        .{ @intCast(cfg.tn), @intCast(cfg.tk) }
    else
        .{ @intCast(cfg.tk), @intCast(cfg.tn) };
}

fn transform0(tb: *mtt.TransformStubBuilder) []const mtt.Value {
    const grid_group = tb.programId(1);
    const m_tile_ids_idx = tb.toIndex(grid_group);
    const m_tile_ids = tb.scalarLoad(tb.prefetchRef(2), &.{m_tile_ids_idx});
    return tb.yield(.{ m_tile_ids, tb.programId(2) });
}

fn transform1(tb: *mtt.TransformStubBuilder) []const mtt.Value {
    const grid_group = tb.programId(1);
    const group_ids_idx = tb.toIndex(grid_group);
    const group_id = tb.scalarLoad(tb.prefetchRef(1), &.{group_ids_idx});
    const group_offset0 = tb.scalarLoad(tb.prefetchRef(4), &.{tb.cIndex(0)});
    const rhs_group = tb.subi(group_id, group_offset0);
    return tb.yield(.{ rhs_group, tb.programId(0), tb.programId(2) });
}

fn transform2(tb: *mtt.TransformStubBuilder) []const mtt.Value {
    const grid_group = tb.programId(1);
    const m_tile_ids_idx = tb.toIndex(grid_group);
    const m_tile_ids = tb.scalarLoad(tb.prefetchRef(2), &.{m_tile_ids_idx});
    return tb.yield(.{ m_tile_ids, tb.programId(0) });
}

fn emitMatmul(
    k: *mtt.Builder,
    cfg: Cfg,
    lhs: mtt.Value,
    rhs: mtt.Value,
    acc_zero: mtt.Value,
) mtt.Value {
    return if (cfg.transpose_rhs)
        k.matmulOpts(lhs, rhs, acc_zero, .{
            .dimension_numbers = k.dotDimensionNumbers(
                &.{1},
                &.{1},
                &.{0},
                &.{0},
                &.{ 0, 0, 1, 0 },
                &.{},
                &.{},
            ),
        })
    else
        k.matmulOpts(lhs, rhs, acc_zero, .{
            .dimension_numbers = k.dotDimensionNumbers(
                &.{1},
                &.{0},
                &.{0},
                &.{1},
                &.{ 0, 0, 1, 1 },
                &.{},
                &.{},
            ),
        });
}

pub fn run(k: *mtt.Builder, cfg: Cfg) mtt.FinishError!void {
    const tm: i64 = @intCast(cfg.tm);
    const tk: i64 = @intCast(cfg.tk);
    const tn: i64 = @intCast(cfg.tn);
    const groups: i64 = @intCast(cfg.num_groups);
    const tiles_m: i64 = @intCast(std.math.divCeil(usize, cfg.m, cfg.tm) catch unreachable);
    const tiles_n: i64 = @intCast(std.math.divCeil(usize, cfg.n, cfg.tn) catch unreachable);
    const tiles_k: i64 = @intCast(std.math.divCeil(usize, cfg.k, cfg.tk) catch unreachable);
    const last_k_tile: i32 = @intCast(tiles_k - 1);
    const k_rem_usize = cfg.k % cfg.tk;
    const k_rem: i32 = @intCast(if (k_rem_usize == 0) cfg.tk else k_rem_usize);
    const metadata_len: i64 = tiles_m + groups - 1;
    const rhs_shape = rhsWindowShape(cfg);
    const share_lhs_zero_with_acc = k_rem_usize != 0 and cfg.dtype == .f32 and cfg.tk == cfg.tn;
    const reuse_rhs_iota_for_store_mask = !cfg.transpose_rhs and tm == rhs_shape[0] and tn == rhs_shape[1];

    const a = try k.declareArgsOpts(.{
        .n_i = .{ .scalar = .i32 },
        .grid_id = .{ .scalar = .i32 },
        .k_i = .{ .scalar = .i32 },

        .group_offsets = .{ .ref = .{ .shape = &.{groups + 1}, .dtype = .i32, .memory_space = .smem, .role = .scalar_prefetch } },
        .group_ids = .{ .ref = .{ .shape = &.{metadata_len}, .dtype = .i32, .memory_space = .smem, .role = .scalar_prefetch } },
        .m_tile_ids = .{ .ref = .{ .shape = &.{metadata_len}, .dtype = .i32, .memory_space = .smem, .role = .scalar_prefetch } },
        .num_active_tiles = .{ .ref = .{ .shape = &.{1}, .dtype = .i32, .memory_space = .smem, .role = .scalar_prefetch } },
        .group_offset = .{ .ref = .{ .shape = &.{1}, .dtype = .i32, .memory_space = .smem, .role = .scalar_prefetch } },

        .lhs = .{ .ref = .{
            .shape = &.{ tm, tk },
            .dtype = cfg.dtype,
            .window = mtt.ArgSpec.WindowSpec{
                .transform_body = transform0,
            },
        } },
        .rhs = .{ .ref = .{
            .shape = &.{ 1, rhs_shape[0], rhs_shape[1] },
            .dtype = cfg.dtype,
            .window = mtt.ArgSpec.WindowSpec{
                .transform_body = transform1,
            },
        } },
        .out = .{ .ref = .{
            .shape = &.{ tm, tn },
            .dtype = cfg.preferred_element_type,
            .role = .output,
            .window = mtt.ArgSpec.WindowSpec{
                .transform_body = transform2,
            },
        } },
        .acc = .{ .ref = .{
            .shape = &.{ tm, tn },
            .dtype = .f32,
            .role = .scratch,
        } },
    }, &.{}, .{
        .dimension_semantics = &.{ .parallel, .arbitrary, .arbitrary },
        .iteration_bounds = &.{ tiles_n, metadata_len, tiles_k },
        .scalar_prefetch = 5,
        .scratch_operands = 1,
        .pallas_window_params = true,
    });

    const c0 = k.cIndex(0);
    const c0_i32 = k.lift(@as(i32, 0));
    const active_tiles = k.scalarLoad(a.num_active_tiles, &.{c0});
    const acc_zero = k.zeros(&.{ tm, tn }, .f32);

    var active_tile_if = k.openIf(k.cmpi(.slt, a.grid_id, active_tiles));
    {
        {
            var zero_acc = k.openIf(k.cmpi(.eq, a.k_i, c0_i32));
            k.vectorStoreAt(a.acc, acc_zero, &.{ c0, c0 });
            zero_acc.yieldThen(.{});
        }

        const last_tile = k.cmpi(.eq, a.k_i, k.lift(last_k_tile));
        var last_tile_if = k.openIfElse(last_tile, .{});
        {
            const lhs_loaded = k.vectorLoadAt(a.lhs, &.{ c0, c0 });
            var lhs_for_dot = lhs_loaded;
            var rhs_iota: ?mtt.Value = null;

            if (k_rem_usize != 0) {
                const lhs_iota = k.iota(&.{ tm, tk }, .i32, &.{1});
                const lhs_mask = k.cmpi(.slt, lhs_iota, k.splat(k_rem, &.{ tm, tk }, .i32));
                lhs_for_dot = k.select(lhs_mask, lhs_loaded, if (share_lhs_zero_with_acc) acc_zero else k.zeros(&.{ tm, tk }, cfg.dtype));
            }

            const rhs_loaded_3d = k.vectorLoadShape(a.rhs, &.{ c0, c0, c0 }, &.{ 1, rhs_shape[0], rhs_shape[1] });
            const rhs_loaded = k.shapeCast(rhs_loaded_3d, &.{ rhs_shape[0], rhs_shape[1] });
            var rhs_for_dot = rhs_loaded;
            if (k_rem_usize != 0) {
                rhs_iota = k.iota(&.{ rhs_shape[0], rhs_shape[1] }, .i32, &.{if (cfg.transpose_rhs) 1 else 0});
                const rhs_mask = k.cmpi(.slt, rhs_iota.?, k.splat(k_rem, &.{ rhs_shape[0], rhs_shape[1] }, .i32));
                rhs_for_dot = k.select(rhs_mask, rhs_loaded, k.zeros(&.{ rhs_shape[0], rhs_shape[1] }, cfg.dtype));
            }

            const acc_loaded = k.vectorLoadAt(a.acc, &.{ c0, c0 });
            const matmul = emitMatmul(k, cfg, lhs_for_dot, rhs_for_dot, acc_zero);
            const acc_next = k.addf(acc_loaded, matmul);
            k.vectorStoreAt(a.acc, acc_next, &.{ c0, c0 });

            const grid_idx = k.toIndex(a.grid_id);
            const group_id = k.scalarLoad(a.group_ids, &.{grid_idx});
            const group_start = k.scalarLoad(a.group_offsets, &.{k.toIndex(group_id)});
            const group_end = k.scalarLoad(a.group_offsets, &.{k.toIndex(k.addi(group_id, k.lift(@as(i32, 1))))});
            const m_tile_id = k.scalarLoad(a.m_tile_ids, &.{grid_idx});
            const m_start = k.muli(m_tile_id, k.lift(@as(i32, @intCast(cfg.tm))));

            const store_iota = if (reuse_rhs_iota_for_store_mask and rhs_iota != null) rhs_iota.? else k.iota(&.{ tm, tn }, .i32, &.{0});
            const store_rows = k.addi(store_iota, k.broadcastTo(m_start, &.{ tm, tn }));
            const lower_ok = k.cmpi(.sge, store_rows, k.broadcastTo(group_start, &.{ tm, tn }));
            const upper_ok = k.cmpi(.slt, store_rows, k.broadcastTo(group_end, &.{ tm, tn }));
            const store_mask = k.andi(lower_ok, upper_ok);

            const acc_stored = k.vectorLoadAt(a.acc, &.{ c0, c0 });
            const out_prev = k.vectorLoadAt(a.out, &.{ c0, c0 });
            const out_prev_f32 = if (cfg.preferred_element_type == .f32) out_prev else out_prev.to(.f32);
            const out_selected = k.select(store_mask, acc_stored, out_prev_f32);
            const out_to_store = if (cfg.preferred_element_type == .f32) out_selected else out_selected.to(cfg.preferred_element_type);
            k.vectorStoreAt(a.out, out_to_store, &.{ c0, c0 });

            last_tile_if.yieldThen(.{});
        }
        {
            const lhs_loaded = k.vectorLoadAt(a.lhs, &.{ c0, c0 });
            const rhs_loaded_3d = k.vectorLoadShape(a.rhs, &.{ c0, c0, c0 }, &.{ 1, rhs_shape[0], rhs_shape[1] });
            const rhs_loaded = k.shapeCast(rhs_loaded_3d, &.{ rhs_shape[0], rhs_shape[1] });
            const acc_loaded = k.vectorLoadAt(a.acc, &.{ c0, c0 });
            const matmul = emitMatmul(k, cfg, lhs_loaded, rhs_loaded, acc_zero);
            k.vectorStoreAt(a.acc, k.addf(acc_loaded, matmul), &.{ c0, c0 });
            last_tile_if.yieldElse(.{});
        }

        active_tile_if.yieldThen(.{});
    }
}

pub const Kernel = mtt_kernel.Kernel(Cfg, .{
    .name = "megablox_gmm_kernel",
    .inputs = &.{ "group_offsets", "group_ids", "m_tile_ids", "num_active_tiles", "group_offset", "lhs", "rhs" },
    .outputs = &.{"out"},
    .run = run,
});
