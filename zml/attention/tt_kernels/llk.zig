//! Tile building blocks of the ported kernels, in the TTKernel DSL: the
//! tt-metal compute API functions without a `ttkernel` operation (as
//! `opaque_call`s, named as tt-metal names them) and the helpers tt-metal's
//! kernels define over them (`mm`, `ew`, ...).

const std = @import("std");

const zml = @import("../../zml.zig");
const dsl = zml.kernel.ttkernel.dsl;
const Builder = dsl.Builder;
const Value = dsl.Value;
const Cb = dsl.CircularBuffer;

const bcast_h = "api/compute/bcast.h";
const binary_h = "api/compute/eltwise_binary.h";
const matmul_h = "api/compute/matmul.h";
const reconfig_h = "api/compute/reconfig_data_format.h";
const copy_h = "api/compute/tile_move_copy.h";
const transpose_h = "api/compute/transpose.h";

// ---------------------------------------------------------------- compute API

pub fn reconfigDataFormat(k: *Builder, srca: Cb, srcb: Cb) void {
    k.call(reconfig_h, "reconfig_data_format", .{}, .{ srca, srcb });
}

pub fn reconfigDataFormatSrca(k: *Builder, srca: Cb) void {
    k.call(reconfig_h, "reconfig_data_format_srca", .{}, .{srca});
}

/// `reconfig_data_format_srca(old, new)`: srcA from `old`'s format to `new`'s.
pub fn reconfigDataFormatSrca2(k: *Builder, old: Cb, new: Cb) void {
    k.call(reconfig_h, "reconfig_data_format_srca", .{}, .{ old, new });
}

pub fn matmulInit(k: *Builder, in0: Cb, in1: Cb, transpose: bool) void {
    k.call(matmul_h, "matmul_init", .{}, .{ in0, in1, @as(u32, @intFromBool(transpose)) });
}

pub const Binary = enum { add, sub, mul };

pub fn binaryInit(k: *Builder, comptime op: Binary, a: Cb, b: Cb) void {
    k.call(binary_h, @tagName(op) ++ "_init", .{}, .{ a, b });
}

pub fn binaryTiles(k: *Builder, comptime op: Binary, a: Cb, b: Cb, ia: anytype, ib: anytype, dst: anytype) void {
    switch (op) {
        .add => k.op(.add_tiles, .{ a, b, ia, ib, dst }),
        .sub => k.op(.sub_tiles, .{ a, b, ia, ib, dst }),
        .mul => k.op(.mul_tiles, .{ a, b, ia, ib, dst }),
    }
}

pub fn copyInit(k: *Builder, cb: Cb) void {
    k.call(copy_h, "copy_init", .{}, .{cb});
}

pub fn transposeInit(k: *Builder, cb: Cb) void {
    k.call(transpose_h, "transpose_init", .{}, .{cb});
}

pub fn transposeTile(k: *Builder, cb: Cb, tile: anytype, dst: anytype) void {
    k.call(transpose_h, "transpose_tile", .{}, .{ cb, tile, dst });
}

/// The broadcasts of `api/compute/bcast.h`: `b`'s first column, first row
/// or first element against every tile of `a`.
pub const Bcast = enum { cols, rows, scalar };

pub fn bcastInit(k: *Builder, comptime op: Binary, comptime how: Bcast, a: Cb, b: Cb) void {
    k.call(bcast_h, @tagName(op) ++ "_bcast_" ++ @tagName(how) ++ "_init", .{}, .{ a, b });
}

pub fn bcastTiles(k: *Builder, comptime op: Binary, comptime how: Bcast, a: Cb, b: Cb, ia: anytype, ib: anytype, dst: anytype) void {
    k.call(bcast_h, @tagName(op) ++ "_tiles_bcast_" ++ @tagName(how), .{}, .{ a, b, ia, ib, dst });
}

// ---------------------------------------------------------------- circular buffers

/// The compute-side circular buffer handshakes, on runtime-picked buffers
/// too (tt-metal's free functions).
const cb_h = "api/compute/cb_api.h";

pub fn wait(k: *Builder, cb: Cb, n: anytype) void {
    if (cb.selected) k.call(cb_h, "cb_wait_front", .{}, .{ cb, n }) else k.op(.cb_wait_front, .{ cb, n });
}

pub fn pop(k: *Builder, cb: Cb, n: anytype) void {
    if (cb.selected) k.call(cb_h, "cb_pop_front", .{}, .{ cb, n }) else k.op(.cb_pop_front, .{ cb, n });
}

pub fn reserve(k: *Builder, cb: Cb, n: anytype) void {
    if (cb.selected) k.call(cb_h, "cb_reserve_back", .{}, .{ cb, n }) else k.op(.cb_reserve_back, .{ cb, n });
}

pub fn push(k: *Builder, cb: Cb, n: anytype) void {
    if (cb.selected) k.call(cb_h, "cb_push_back", .{}, .{ cb, n }) else k.op(.cb_push_back, .{ cb, n });
}

/// Packs destination register 0 into tile `index` of `o`, between the
/// register handshakes, after `body` computed it.
fn packOne(k: *Builder, o: Cb, index: anytype, body: anytype) void {
    k.op(.tile_regs_acquire, .{});
    body.emit(k);
    k.op(.tile_regs_commit, .{});
    k.op(.tile_regs_wait, .{});
    k.op(.pack_tile, .{ 0, o, index });
    k.op(.tile_regs_release, .{});
}

/// `for i in 0..n` with `i` as an `i32`.
pub fn range(k: *Builder, n: anytype) struct { dsl.ForScope(0), Value } {
    const loop = k.openFor(0, n, 1, .{});
    return .{ loop, loop.iv.toI32() };
}

// ---------------------------------------------------------------- tt-metal helpers

/// `o[Mt, Nt] = a[Mt, Kt] @ (transposed ? b[Nt, Kt]^T : b[Kt, Nt])`, the
/// inputs waited for (`mm` of the GDN kernels).
pub fn mm(k: *Builder, a: Cb, b: Cb, o: Cb, mt: u32, kt: u32, nt: u32, transposed: bool) void {
    reserve(k, o, mt * nt);
    k.op(.pack_reconfig_data_format, .{o});
    // matmul_tiles(a, b): a to srcB, b to srcA.
    reconfigDataFormat(k, b, a);
    matmulInit(k, a, b, transposed);
    var m_loop, const mi = range(k, mt);
    {
        var n_loop, const ni = range(k, nt);
        {
            k.op(.tile_regs_acquire, .{});
            var k_loop, const ki = range(k, kt);
            {
                const bi = if (transposed) ni.mul(kt).add(ki) else ki.mul(nt).add(ni);
                k.op(.matmul_tiles, .{ a, b, mi.mul(kt).add(ki), bi, 0 });
            }
            k_loop.yield(.{});
            k.op(.tile_regs_commit, .{});
            k.op(.tile_regs_wait, .{});
            k.op(.pack_tile, .{ 0, o, mi.mul(nt).add(ni) });
            k.op(.tile_regs_release, .{});
        }
        n_loop.yield(.{});
    }
    m_loop.yield(.{});
    push(k, o, mt * nt);
}

/// `o = a op b` over `n` tiles (`ew`).
pub fn ew(k: *Builder, a: Cb, b: Cb, o: Cb, n: u32, comptime op: Binary) void {
    reserve(k, o, n);
    k.op(.pack_reconfig_data_format, .{o});
    reconfigDataFormat(k, a, b);
    binaryInit(k, op, a, b);
    var loop, const i = range(k, n);
    packOne(k, o, i, struct {
        a: Cb,
        b: Cb,
        i: Value,
        fn emit(s: @This(), kk: *Builder) void {
            binaryTiles(kk, op, s.a, s.b, s.i, s.i, 0);
        }
    }{ .a = a, .b = b, .i = i });
    loop.yield(.{});
    push(k, o, n);
}

/// `o = a op b` on single tiles `a[ia]`, `b[ib]` into `o[0]` (`ewt`).
pub fn ewt(k: *Builder, a: Cb, ia: u32, b: Cb, ib: u32, o: Cb, comptime op: Binary) void {
    reserve(k, o, 1);
    k.op(.pack_reconfig_data_format, .{o});
    reconfigDataFormat(k, a, b);
    binaryInit(k, op, a, b);
    packOne(k, o, 0, struct {
        a: Cb,
        b: Cb,
        ia: u32,
        ib: u32,
        fn emit(s: @This(), kk: *Builder) void {
            binaryTiles(kk, op, s.a, s.b, s.ia, s.ib, 0);
        }
    }{ .a = a, .b = b, .ia = ia, .ib = ib });
    push(k, o, 1);
}

/// `o[Mt, Nt] = a[Mt, Nt] op b`, `b` broadcast: column tile `mi` of `b`
/// (`.cols`), row tile `ni` (`.rows`), or its first tile (`.scalar`).
pub fn bcast(k: *Builder, a: Cb, b: Cb, o: Cb, mt: u32, nt: u32, comptime op: Binary, comptime how: Bcast) void {
    reserve(k, o, mt * nt);
    k.op(.pack_reconfig_data_format, .{o});
    reconfigDataFormat(k, a, b);
    bcastInit(k, op, how, a, b);
    var m_loop, const mi = range(k, mt);
    {
        var n_loop, const ni = range(k, nt);
        const i = mi.mul(nt).add(ni);
        packOne(k, o, i, struct {
            a: Cb,
            b: Cb,
            i: Value,
            ib: Value,
            fn emit(s: @This(), kk: *Builder) void {
                bcastTiles(kk, op, how, s.a, s.b, s.i, s.ib, 0);
            }
        }{ .a = a, .b = b, .i = i, .ib = switch (how) {
            .cols => mi,
            .rows => ni,
            .scalar => k.int(0),
        } });
        n_loop.yield(.{});
    }
    m_loop.yield(.{});
    push(k, o, mt * nt);
}

/// `o = exp(in)` over `n` tiles (`expc`).
pub fn exp(k: *Builder, in: Cb, o: Cb, n: u32) void {
    reserve(k, o, n);
    k.op(.pack_reconfig_data_format, .{o});
    reconfigDataFormatSrca(k, in);
    copyInit(k, in);
    k.op(.exp_tile_init, .{});
    var loop, const i = range(k, n);
    packOne(k, o, i, struct {
        in: Cb,
        i: Value,
        fn emit(s: @This(), kk: *Builder) void {
            kk.op(.copy_tile, .{ s.in, s.i, 0 });
            kk.op(.exp_tile, .{0});
        }
    }{ .in = in, .i = i });
    loop.yield(.{});
    push(k, o, n);
}

/// `o[cols, rows]`, tile `[c, r]` the transpose of tile `[r, c]` of
/// `in[rows, cols]` (`transpose_col` for one column, `k_dec_t`).
pub fn transposeBlock(k: *Builder, in: Cb, o: Cb, rows: u32, cols: u32) void {
    reserve(k, o, rows * cols);
    k.op(.pack_reconfig_data_format, .{o});
    reconfigDataFormatSrca(k, in);
    transposeInit(k, in);
    var c_loop, const c = range(k, cols);
    {
        var r_loop, const r = range(k, rows);
        packOne(k, o, c.mul(rows).add(r), struct {
            in: Cb,
            i: Value,
            fn emit(s: @This(), kk: *Builder) void {
                transposeTile(kk, s.in, s.i, 0);
            }
        }{ .in = in, .i = r.mul(cols).add(c) });
        r_loop.yield(.{});
    }
    c_loop.yield(.{});
    push(k, o, rows * cols);
}

pub const TileRef = struct { cb: Cb, tile: u32 };

/// `o[i] = tiles[i]`: tiles of several buffers copied into one (`asm4`).
pub fn assemble(k: *Builder, tiles: []const TileRef, o: Cb) void {
    reserve(k, o, @as(u32, @intCast(tiles.len)));
    k.op(.pack_reconfig_data_format, .{o});
    for (tiles, 0..) |t, i| {
        reconfigDataFormatSrca(k, t.cb);
        copyInit(k, t.cb);
        packOne(k, o, @as(u32, @intCast(i)), struct {
            t: TileRef,
            fn emit(s: @This(), kk: *Builder) void {
                kk.op(.copy_tile, .{ s.t.cb, s.t.tile, 0 });
            }
        }{ .t = t });
    }
    push(k, o, @as(u32, @intCast(tiles.len)));
}

/// `o[0] = src[tile]` (`cpy_t`).
pub fn copyTile(k: *Builder, src: Cb, tile: anytype, o: Cb) void {
    reserve(k, o, 1);
    k.op(.pack_reconfig_data_format, .{o});
    reconfigDataFormatSrca(k, src);
    copyInit(k, src);
    packOne(k, o, 0, struct {
        src: Cb,
        tile: Value,
        fn emit(s: @This(), kk: *Builder) void {
            kk.op(.copy_tile, .{ s.src, s.tile, 0 });
        }
    }{ .src = src, .tile = k.lift(tile) });
    push(k, o, 1);
}

// ---------------------------------------------------------------- data movement

/// Reads tiles `base + t` of `acc` into consecutive pages of `cb`, `t` in
/// `0..n`: `n` pages reserved, read, pushed.
pub fn readTiles(k: *Builder, acc: Value, cb: Cb, base: Value, n: u32, tile_bytes: Value) void {
    k.op(.cb_reserve_back, .{ cb, n });
    const l1 = k.op(.get_write_ptr, .{cb});
    var loop, const t = range(k, n);
    k.op(.noc_async_read_tile, .{ base.add(t), acc, l1.add(t.mul(tile_bytes)) });
    loop.yield(.{});
    k.op(.noc_async_read_barrier, .{});
    k.op(.cb_push_back, .{ cb, n });
}

/// Reads `rows` rows of `cols` tiles of `acc`, row `r` from tile
/// `base + r * stride`, into consecutive pages of `cb`.
pub fn readBlock(k: *Builder, acc: Value, cb: Cb, base: Value, rows: u32, cols: u32, stride: anytype, tile_bytes: Value) void {
    k.op(.cb_reserve_back, .{ cb, rows * cols });
    const l1 = k.op(.get_write_ptr, .{cb});
    var r_loop, const r = range(k, rows);
    {
        var c_loop, const c = range(k, cols);
        const page = r.mul(cols).add(c);
        k.op(.noc_async_read_tile, .{ base.add(r.mul(stride)).add(c), acc, l1.add(page.mul(tile_bytes)) });
        c_loop.yield(.{});
    }
    r_loop.yield(.{});
    k.op(.noc_async_read_barrier, .{});
    k.op(.cb_push_back, .{ cb, rows * cols });
}

/// Writes the `rows * cols` front pages of `cb` to tiles of `acc`, row `r`
/// at tile `base + r * stride`: waited for, written, popped.
pub fn writeBlock(k: *Builder, acc: Value, cb: Cb, base: Value, rows: u32, cols: u32, stride: anytype, tile_bytes: Value) void {
    k.op(.cb_wait_front, .{ cb, rows * cols });
    const l1 = k.op(.get_read_ptr, .{cb});
    var r_loop, const r = range(k, rows);
    {
        var c_loop, const c = range(k, cols);
        const page = r.mul(cols).add(c);
        k.op(.noc_async_write_tile, .{ base.add(r.mul(stride)).add(c), acc, l1.add(page.mul(tile_bytes)) });
        c_loop.yield(.{});
    }
    r_loop.yield(.{});
    k.op(.noc_async_write_barrier, .{});
    k.op(.cb_pop_front, .{ cb, rows * cols });
}

/// Writes the `n` front pages of `cb` to tiles `base + t` of `acc`.
pub fn writeTiles(k: *Builder, acc: Value, cb: Cb, base: Value, n: u32, tile_bytes: Value) void {
    writeBlock(k, acc, cb, base, 1, n, n, tile_bytes);
}

/// The program slot of a kernel: its processor, and so its NoC.
pub const Slot = enum { reader, writer, compute };

/// The C++ source of the kernel `body(k, cfg)` builds for `slot`: its
/// transfers on the slot's NoC (the reader's 0, the writer's 1).
pub fn source(allocator: std.mem.Allocator, ctx: anytype, comptime name: []const u8, slot: Slot, comptime body: anytype, cfg: anytype) ![]u8 {
    var k = try Builder.open(allocator, ctx, name, if (slot == .compute) .compute else .noc);
    defer k.deinit();
    k.default_noc = if (slot == .writer) 1 else 0;
    body(&k, cfg);
    try k.finish();
    return k.cpp(allocator);
}

// ---------------------------------------------------------------- synchronization

const dataflow_h = "api/dataflow/dataflow_api.h";

/// Copies `size` bytes from this core's L1 at `src` to its L1 at `dst`
/// through the kernel's NoC: the core's own address on it. Waits for it.
pub fn localCopy(k: *Builder, src: Value, dst: Value, size: anytype) void {
    const noc = k.kernelNoc();
    const src_noc = k.callValue(dataflow_h, "get_noc_addr", .{}, .{ src, noc }, .int(k.ctx, .i64));
    k.call(dataflow_h, "noc_async_read", .{}, .{ src_noc, dst, size, noc });
    k.op(.noc_async_read_barrier, .{noc});
}

/// tt-metal's `common_values.hpp` semaphore states.
pub const invalid = 0;
pub const valid = 1;

/// A program semaphore of a data movement kernel, its id compile-time
/// argument `ct_index` (tt-metal's `Semaphore<>`), signalled on the
/// kernel's NoC.
pub const Semaphore = struct {
    k: *Builder,
    /// L1 address.
    addr: Value,
    ptr: Value,
    noc: Value,

    pub fn init(k: *Builder, ct_index: u32) Semaphore {
        const addr = k.op(.get_semaphore, .{k.compileTimeArg(ct_index)});
        return .{ .k = k, .addr = addr, .ptr = k.l1Ptr(addr, 32), .noc = k.kernelNoc() };
    }

    pub fn wait(self: Semaphore, v: anytype) void {
        self.k.call(dataflow_h, "noc_semaphore_wait", .{}, .{ self.ptr, v });
    }

    pub fn waitMin(self: Semaphore, v: anytype) void {
        self.k.call(dataflow_h, "noc_semaphore_wait_min", .{}, .{ self.ptr, v });
    }

    pub fn set(self: Semaphore, v: anytype) void {
        self.k.op(.noc_semaphore_set, .{ self.ptr, v });
    }

    pub fn value(self: Semaphore) Value {
        return self.k.loadL1(self.ptr, 0, 32);
    }

    /// This semaphore's value written to the same semaphore of the cores of
    /// the rectangle `(x0, y0)..(x1, y1)` (NoC coordinates), `n` of them.
    pub fn setMulticast(self: Semaphore, x0: Value, y0: Value, x1: Value, y1: Value, n: anytype) void {
        const dst = self.k.op(.get_noc_multicast_addr, .{ x0, y0, x1, y1, self.addr, self.noc });
        self.k.op(.noc_semaphore_set_multicast, .{ self.addr, dst, n });
    }

    /// Adds `v` to the same semaphore of core `(x, y)`.
    pub fn up(self: Semaphore, x: Value, y: Value, v: anytype) void {
        const remote = self.k.op(.get_noc_addr, .{ x, y, self.addr, self.noc });
        self.k.op(.noc_semaphore_inc, .{ remote, v, self.noc });
    }
};

/// The compute threads of a core: unpack, math, pack (`ckernel::ThreadId`).
pub const ThreadId = enum(u8) { brisc = 0, unpack = 1, math = 2, pack = 3 };

/// Posts `v` to `thread`'s mailbox (`ckernel::mailbox_write`).
pub fn mailboxWrite(k: *Builder, thread: ThreadId, v: Value) void {
    k.call("ckernel.h", "ckernel::mailbox_write", .{}, .{ @as(u8, @intFromEnum(thread)), v });
}

/// Waits for and takes the value `thread` posted to this compute thread
/// (`ckernel::mailbox_read`): each compute thread reads its own mailbox.
pub fn mailboxRead(k: *Builder, thread: ThreadId) Value {
    return k.callValue("ckernel.h", "ckernel::mailbox_read", .{}, .{@as(u8, @intFromEnum(thread))}, .int(k.ctx, .i32));
}
