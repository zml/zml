//! tt-metal's chunked Gated Delta Rule, ported: the host program factories
//! of `chunk_gdn_prep` and `chunk_gdn_scan` (`ttnn/cpp/ttnn/operations/
//! transformer/chunk_gated_delta_rule/device/chunk_gdn_phased_program_factory.cpp`,
//! tt-metal d04395ed) with their device kernels, and the preparation
//! `ttnn.transformer.chunk_gated_delta_rule` does around them. Queries, keys
//! and values are read in place, token-major (the op's "flat" inputs), as
//! Tenstorrent's Qwen3.6 model runs it.
//!
//! The prep phase computes, per chunk of every head, the WY-form inputs of
//! the recurrence (decayed keys and queries, intra-chunk attention, the
//! inverse of the unit lower-triangular system); the scan phase runs the
//! sequential state update over the chunks of each head, split over value
//! columns. Accumulation is fp32 (HiFi4).

const std = @import("std");

const Compiler = @import("../../Compiler.zig");
const zml = @import("../../zml.zig");
const dsl = zml.kernel.ttkernel.dsl;
const program = dsl.program;
const kernels = @import("gated_delta_net_kernels.zig");
const llk = @import("llk.zig");
const Tensor = zml.Tensor;

const cb = kernels.cb;

pub const Options = struct {
    /// Tokens per chunk, a multiple of 32. Tenstorrent's models use 32: at 64
    /// the inverse of the 2x2-tile blocks loses precision.
    chunk_size: u32 = 32,
    /// `(cols, rows)` of the cores available, from (0, 0).
    grid: [2]u32 = .{ 8, 8 },
};

const C = program.Arg;

fn val(x: u32) C {
    return .{ .value = x };
}

const compute_config: program.ComputeConfig = .{ .math_fidelity = .hifi4, .fp32_dest_acc_en = true, .math_approx_mode = false };

const Cbs = struct {
    list: std.ArrayList(program.CircularBuffer) = .empty,
    cores: program.CoreRange,
    more_cores: []const program.CoreRange,

    fn add(self: *Cbs, a: std.mem.Allocator, index: u32, tiles: u32, buffers: u32, dtype: program.DataType) !void {
        const ts = dtype.tileSize();
        try self.list.append(a, .{ .index = index, .dtype = dtype, .page_size = ts, .num_pages = tiles * buffers, .cores = self.cores, .more_cores = self.more_cores });
    }
};

/// Compile-time arguments of a DSL kernel: its circular buffers, then the
/// accessor arguments of the io-tensors `tensors`.
fn ctArgs(allocator: std.mem.Allocator, cb_ids: []const u32, tensors: []const u32) ![]const C {
    const out = try allocator.alloc(C, cb_ids.len + tensors.len);
    for (out[0..cb_ids.len], cb_ids) |*a, id| a.* = .{ .cb = id };
    for (out[cb_ids.len..], tensors) |*a, t| a.* = .{ .tensor_accessor_args = t };
    return out;
}

fn coreAt(i: u32, cols: u32) program.CoreCoord {
    return .{ .x = i % cols, .y = i / cols };
}

/// The prep program (`ChunkGdnPrepProgramFactory`) over token-major
/// inputs, read in place (its "flat" q/k/v), the queries and keys already
/// normalized. io-tensors: `q`, `k` (`[B, T, hk * K]` bf16), `v`
/// (`[B, T, hv * V]` bf16) with `T = NC * C`, `g`, `beta` (`[B * hv, NC, C,
/// 1]` f32), `eye`, `tril`, `ones` (`[1, 1, C, C]` f32), `masks` (`[1, 1,
/// 32, 96]` f32), then the outputs `v_beta`, `kd`, `q_decay`, `intra`,
/// `k_dec_t`, `dl`, `t_inv`. Value head `h` reads key head `h / (hv / hk)`.
pub fn prepProgram(allocator: std.mem.Allocator, batch: u32, hv: u32, hk: u32, nc: u32, chunk_size: u32, key_dim: u32, val_dim: u32, grid: [2]u32) ![]u8 {
    const bh = batch * hv;
    const ct = chunk_size / 32;
    const kt = key_dim / 32;
    const vt = val_dim / 32;
    const cc = ct * ct;
    const ck = ct * kt;
    const cv = ct * vt;
    const kv = kt * vt;
    const kc = kt * ct;
    const scr = @max(cc, ck, cv, kv, kc);

    // Work items (head, chunk) spread over the first cores, row-major.
    const total = bh * nc;
    const used = @min(total, grid[0] * grid[1]);
    const base = total / used;
    const rem = total % used;
    const ranges = program.firstCores(used, grid[0]);
    const more: []const program.CoreRange = if (ranges[1]) |r| try allocator.dupe(program.CoreRange, &.{r}) else &.{};

    var cbs: Cbs = .{ .cores = ranges[0], .more_cores = more };
    try cbs.add(allocator, cb.q, ck, 1, .bf16);
    try cbs.add(allocator, cb.k, ck, 1, .bf16);
    try cbs.add(allocator, cb.v, cv, 1, .bf16);
    try cbs.add(allocator, cb.g, ct, 1, .f32);
    try cbs.add(allocator, cb.beta, ct, 1, .f32);
    try cbs.add(allocator, cb.eye, cc, 1, .f32);
    try cbs.add(allocator, cb.tril, cc, 1, .f32);
    try cbs.add(allocator, cb.ones, cc, 1, .f32);
    try cbs.add(allocator, cb.s, kv, 2, .f32);
    try cbs.add(allocator, cb.decay, ct, 1, .f32);
    try cbs.add(allocator, cb.decay_exp, ct, 1, .f32);
    try cbs.add(allocator, cb.decayfac, ct, 1, .f32);
    try cbs.add(allocator, cb.lmask, cc, 1, .f32);
    try cbs.add(allocator, cb.tinv, cc, 1, .f32);
    try cbs.add(allocator, cb.vbeta, cv, 1, .f32);
    try cbs.add(allocator, cb.kbeta, ck, 1, .f32);
    try cbs.add(allocator, cb.out, cv, 2, .bf16);
    // The prep kernel loads the three quadrant masks into `u`: tt-metal's
    // `cv` tiles fall short of them below 128 value dims at 32-step chunks.
    try cbs.add(allocator, cb.u, @max(cv, 3), 1, .f32);
    try cbs.add(allocator, cb.w, ck, 1, .f32);
    try cbs.add(allocator, cb.qdecay, ck, 1, .f32);
    try cbs.add(allocator, cb.intra, cc, 1, .f32);
    try cbs.add(allocator, cb.s2, kv, 2, .f32);
    try cbs.add(allocator, cb.vnew, cv, 1, .f32);
    try cbs.add(allocator, cb.ointer, cv, 1, .f32);
    try cbs.add(allocator, cb.kdec_t, kc, 1, .f32);
    try cbs.add(allocator, cb.supd, kv, 1, .f32);
    try cbs.add(allocator, cb.stmp, kv, 1, .f32);
    try cbs.add(allocator, cb.final_s, kv, 1, .f32);
    try cbs.add(allocator, cb.scr1, scr, 1, .f32);
    try cbs.add(allocator, cb.scr2, scr, 1, .f32);
    try cbs.add(allocator, cb.scr3, scr, 1, .f32);
    try cbs.add(allocator, cb.s3, kv, 2, .f32);

    const reader_ct = try ctArgs(allocator, &kernels.cbIds(&kernels.prep_reader_cbs), &.{ 0, 1, 2, 3, 4, 5, 6, 7, 8 });
    const writer_ct = try ctArgs(allocator, &kernels.cbIds(&kernels.prep_writer_cbs), &.{ 9, 10, 11, 12, 13, 14, 15 });
    const compute_ct = try ctArgs(allocator, &kernels.cbIds(&kernels.prep_compute_cbs), &.{});

    const reader_rt = try allocator.alloc(program.CoreArgs, used);
    const writer_rt = try allocator.alloc(program.CoreArgs, used);
    const compute_rt = try allocator.alloc(program.CoreArgs, used);
    var start: u32 = 0;
    for (0..used) |ii| {
        const i: u32 = @intCast(ii);
        const core = coreAt(i, grid[0]);
        const count = base + @intFromBool(i < rem);
        reader_rt[ii] = .{ .core = core, .args = try allocator.dupe(C, &.{
            val(start),               val(count),               .{ .tensor_address = 0 }, .{ .tensor_address = 1 },
            .{ .tensor_address = 2 }, .{ .tensor_address = 3 }, .{ .tensor_address = 4 }, .{ .tensor_address = 5 },
            .{ .tensor_address = 6 }, .{ .tensor_address = 7 }, .{ .tensor_address = 8 }, val(nc),
            val(hv),                  val(hk),
        }) };
        writer_rt[ii] = .{ .core = core, .args = try allocator.dupe(C, &.{
            val(start),                val(count),                .{ .tensor_address = 9 },  .{ .tensor_address = 10 },
            .{ .tensor_address = 11 }, .{ .tensor_address = 12 }, .{ .tensor_address = 13 }, .{ .tensor_address = 14 },
            .{ .tensor_address = 15 },
        }) };
        compute_rt[ii] = .{ .core = core, .args = try allocator.dupe(C, &.{val(count)}) };
        start += count;
    }

    const ctx = try dsl.newContext();
    defer ctx.deinit();
    const cfg: kernels.PrepConfig = .{ .ct = ct, .kt = kt, .vt = vt };
    const p: program.Program = .{
        .kernels = &.{
            .{ .thread = .reader, .source = try llk.source(allocator, ctx, "chunk_gdn_prep_reader", .reader, kernels.prepReader, cfg), .cores = ranges[0], .more_cores = more, .compile_time_args = reader_ct, .runtime_args = reader_rt },
            .{ .thread = .writer, .source = try llk.source(allocator, ctx, "chunk_gdn_prep_writer", .writer, kernels.prepWriter, cfg), .cores = ranges[0], .more_cores = more, .compile_time_args = writer_ct, .runtime_args = writer_rt },
            .{ .thread = .{ .compute = compute_config }, .source = try llk.source(allocator, ctx, "chunk_gdn_prep", .compute, kernels.prepCompute, cfg), .cores = ranges[0], .more_cores = more, .compile_time_args = compute_ct, .runtime_args = compute_rt },
        },
        .circular_buffers = cbs.list.items,
    };
    return p.artifact(allocator);
}

/// The scan program (`ChunkGdnScanProgramFactory`). io-tensors: the seven
/// prep outputs, the initial state (`[BH, K, V]` f32), then the outputs `o`
/// (`[BH, NC, C, V]` f32) and the final state (`[BH, K, V]` f32).
pub fn scanProgram(allocator: std.mem.Allocator, bh: u32, nc: u32, chunk_size: u32, key_dim: u32, val_dim: u32, grid: [2]u32) ![]u8 {
    const ct = chunk_size / 32;
    const kt = key_dim / 32;
    const vt_full = val_dim / 32;
    const ncores = grid[0] * grid[1];
    if (bh > ncores) return error.TooManyHeads;
    // Value columns per head split into `nv` blocks, one core each.
    var nv: u32 = vt_full;
    while (nv > 1) : (nv -= 1) {
        if (vt_full % nv == 0 and bh * nv <= ncores) break;
    }
    const vt = vt_full / nv;
    const used = bh * nv;
    const cc = ct * ct;
    const ck = ct * kt;
    const cv = ct * vt;
    const kv = kt * vt;
    const kc = kt * ct;
    const scr = @max(cc, ck, cv, kv, kc);

    const ranges = program.firstCores(used, grid[0]);
    const more: []const program.CoreRange = if (ranges[1]) |r| try allocator.dupe(program.CoreRange, &.{r}) else &.{};
    var cbs: Cbs = .{ .cores = ranges[0], .more_cores = more };
    try cbs.add(allocator, cb.u, cv, 1, .f32);
    try cbs.add(allocator, cb.w, ck, 1, .f32);
    try cbs.add(allocator, cb.qdecay, ck, 1, .f32);
    try cbs.add(allocator, cb.intra, cc, 1, .f32);
    try cbs.add(allocator, cb.kdec_t, kc, 1, .f32);
    try cbs.add(allocator, cb.decayfac, 1, 1, .f32);
    try cbs.add(allocator, cb.tinv, cc, 1, .f32);
    try cbs.add(allocator, cb.s, kv, 1, .f32);
    try cbs.add(allocator, cb.s2, kv, 1, .f32);
    try cbs.add(allocator, cb.s3, kv, 1, .f32);
    try cbs.add(allocator, cb.out, cv, 2, .f32);
    try cbs.add(allocator, cb.final_s, kv, 1, .f32);
    try cbs.add(allocator, cb.vnew, cv, 1, .f32);
    try cbs.add(allocator, cb.ointer, cv, 1, .f32);
    try cbs.add(allocator, cb.supd, kv, 1, .f32);
    try cbs.add(allocator, cb.stmp, kv, 1, .f32);
    try cbs.add(allocator, cb.scr1, scr, 1, .f32);

    const reader_ct = try ctArgs(allocator, &kernels.cbIds(&kernels.scan_reader_cbs), &.{ 0, 1, 2, 3, 4, 5, 6, 7 });
    const writer_ct = try ctArgs(allocator, &kernels.cbIds(&kernels.scan_writer_cbs), &.{ 8, 9 });
    const compute_ct = try ctArgs(allocator, &kernels.cbIds(&kernels.scan_compute_cbs), &.{});

    const reader_rt = try allocator.alloc(program.CoreArgs, used);
    const writer_rt = try allocator.alloc(program.CoreArgs, used);
    const compute_rt = try allocator.alloc(program.CoreArgs, used);
    for (0..used) |ii| {
        const i: u32 = @intCast(ii);
        const core = coreAt(i, grid[0]);
        const h = i / nv;
        const vb = i % nv;
        reader_rt[ii] = .{ .core = core, .args = try allocator.dupe(C, &.{
            val(h),                   val(vb),                  val(nc),                  .{ .tensor_address = 0 },
            .{ .tensor_address = 1 }, .{ .tensor_address = 2 }, .{ .tensor_address = 3 }, .{ .tensor_address = 4 },
            .{ .tensor_address = 5 }, .{ .tensor_address = 6 }, .{ .tensor_address = 7 },
        }) };
        writer_rt[ii] = .{ .core = core, .args = try allocator.dupe(C, &.{ val(h), val(vb), val(nc), .{ .tensor_address = 8 }, .{ .tensor_address = 9 } }) };
        compute_rt[ii] = .{ .core = core, .args = try allocator.dupe(C, &.{val(nc)}) };
    }

    const ctx = try dsl.newContext();
    defer ctx.deinit();
    const cfg: kernels.ScanConfig = .{ .ct = ct, .kt = kt, .vt = vt, .vt_full = vt_full };
    const p: program.Program = .{
        .kernels = &.{
            .{ .thread = .reader, .source = try llk.source(allocator, ctx, "chunk_gdn_scan_reader", .reader, kernels.scanReader, cfg), .cores = ranges[0], .more_cores = more, .compile_time_args = reader_ct, .runtime_args = reader_rt },
            .{ .thread = .writer, .source = try llk.source(allocator, ctx, "chunk_gdn_scan_writer", .writer, kernels.scanWriter, cfg), .cores = ranges[0], .more_cores = more, .compile_time_args = writer_ct, .runtime_args = writer_rt },
            .{ .thread = .{ .compute = compute_config }, .source = try llk.source(allocator, ctx, "chunk_gdn_scan", .compute, kernels.scanCompute, cfg), .cores = ranges[0], .more_cores = more, .compile_time_args = compute_ct, .runtime_args = compute_rt },
        },
        .circular_buffers = cbs.list.items,
    };
    return p.artifact(allocator);
}

/// The constants of the prep phase (`build_const_tiles`): `[1, 1, C, C]`
/// identity, lower-triangular ones and all ones, and the three 32x32
/// quadrant masks of the 16x16 sub-blocked inverse (top-left, bottom-right,
/// bottom-left) as one `[1, 1, 32, 96]` row of tiles.
fn constants(allocator: std.mem.Allocator, chunk_size: u32) [4]Tensor {
    const n: usize = chunk_size;
    const eye = allocator.alloc(f32, n * n) catch @panic("OOM");
    const tril = allocator.alloc(f32, n * n) catch @panic("OOM");
    const ones = allocator.alloc(f32, n * n) catch @panic("OOM");
    for (0..n) |i| for (0..n) |j| {
        eye[i * n + j] = if (i == j) 1 else 0;
        tril[i * n + j] = if (j <= i) 1 else 0;
        ones[i * n + j] = 1;
    };
    const masks = allocator.alloc(f32, 32 * 96) catch @panic("OOM");
    for (0..32) |i| for (0..32) |j| {
        const lo_i = i < 16;
        const lo_j = j < 16;
        masks[i * 96 + j] = if (lo_i and lo_j) 1 else 0;
        masks[i * 96 + 32 + j] = if (!lo_i and !lo_j) 1 else 0;
        masks[i * 96 + 64 + j] = if (!lo_i and lo_j) 1 else 0;
    };
    const cc: zml.Shape = .init(.{ 1, 1, chunk_size, chunk_size }, .f32);
    return .{
        .constantTensor(cc, std.mem.sliceAsBytes(eye)),
        .constantTensor(cc, std.mem.sliceAsBytes(tril)),
        .constantTensor(cc, std.mem.sliceAsBytes(ones)),
        .constantTensor(.init(.{ 1, 1, 32, 96 }, .f32), std.mem.sliceAsBytes(masks)),
    };
}

/// Whether the kernels take these queries and values: whole tiles of key
/// and value dims, at most as many batches times heads as default cores.
pub fn supports(queries: Tensor, values: Tensor) bool {
    const opts: Options = .{};
    const b = if (queries.shape().hasTag(.b) != null) queries.dim(.b) else 1;
    return @mod(queries.dim(.k), 32) == 0 and @mod(values.dim(.v), 32) == 0 and b * queries.dim(.h) <= opts.grid[0] * opts.grid[1];
}

/// Gated Delta Net over a sequence with tt-metal's chunked kernels, with
/// the semantics of `zml.nn.GatedDeltaNet.forward`: `queries`, `keys`
/// (`.{ .s, .h, .k }`), `values` (`.{ .s, .h, .v }`), the per-step decay
/// multipliers `alphas` and gates `betas` (`.{ .s, .h }`), the state
/// (`.{ .h, .v, .k }`), all with an optional `.b` axis; steps from
/// `active_length` on leave the state alone and output zeros. The chunked
/// form stays accurate for L2-normalized keys, as GDN models use. At most
/// as many batches times heads as cores.
pub fn gatedDeltaNet(queries: Tensor, keys: Tensor, values: Tensor, alphas: Tensor, betas: Tensor, initial_state: zml.nn.GatedDeltaNet.Input, active_length: Tensor, opts: Options) zml.nn.GatedDeltaNet.Output {
    const allocator = Compiler.current().arena.allocator();
    const had_b = queries.shape().hasTag(.b) != null;
    const lift = struct {
        fn f(t: Tensor, order: anytype, batched: bool) Tensor {
            return (if (batched) t else t.insertAxes(0, .{.b})).transpose(order);
        }
    }.f;
    const q = lift(queries, .{ .b, .s, .h, .k }, had_b);
    const k = lift(keys, .{ .b, .s, .h, .k }, had_b);
    const v = lift(values, .{ .b, .s, .h, .v }, had_b);
    const alpha = lift(alphas, .{ .b, .s, .h }, had_b);
    const beta_in = lift(betas, .{ .b, .s, .h }, had_b);
    const s0 = lift(initial_state.s, .{ .b, .h, .k, .v }, had_b);

    const b = q.dim(.b);
    const s = q.dim(.s);
    const h = q.dim(.h);
    const kd = q.dim(.k);
    const vd = v.dim(.v);
    const csz: i64 = opts.chunk_size;
    const len = std.mem.alignForward(i64, s, csz);
    const nc = @divExact(len, csz);

    // Steps past `active_length` keep the state: no update (beta 0), no
    // decay (log-decay 0).
    const step = Tensor.iota(alpha.shape().withDtype(.i32), .s);
    const active = step.cmp(.LT, active_length.convert(.i32).broad(step.shape()));
    const zero = Tensor.scalar(0, .f32).broad(alpha.shape().withDtype(.f32));
    const g = active.select(alpha.convert(.f32).log(), zero);
    const beta = active.select(beta_in.convert(.f32), zero);

    // Queries, keys and values token-major, `[b, steps, heads * dim]`, as
    // they are laid out: the kernels read each head's chunks in place.
    const tokenMajor = struct {
        fn f(t: Tensor, pad_to: i64) Tensor {
            const p = t.convert(.bf16).pad(0, .{ .s = Tensor.Pad{ .high = pad_to - t.dim(.s) } });
            return p.reshape(.{ p.dim(0), pad_to, p.dim(2) * p.dim(3) });
        }
    }.f;
    // Gates head-major, `[b * h, chunks, chunk, 1]`.
    const headMajor = struct {
        fn f(t: Tensor, n_chunks: i64, chunk: i64) Tensor {
            const p = t.transpose(.{ .b, .h, .s }).pad(0, .{ .s = Tensor.Pad{ .high = n_chunks * chunk - t.dim(.s) } });
            return p.reshape(.{ p.dim(.b) * p.dim(.h), n_chunks, chunk, 1 });
        }
    }.f;
    const consts = constants(allocator, opts.chunk_size);
    const bh = b * h;

    const prep_art = prepProgram(allocator, @intCast(b), @intCast(h), @intCast(h), @intCast(nc), opts.chunk_size, @intCast(kd), @intCast(vd), opts.grid) catch |err|
        std.debug.panic("gatedDeltaNet prep: {}", .{err});
    const f32s = struct {
        fn shape(dims: anytype) zml.Shape {
            return zml.Shape.init(dims, .f32);
        }
    }.shape;
    const prep = zml.ops.ttKernel(&.{
        tokenMajor(q, len), tokenMajor(k, len), tokenMajor(v, len), headMajor(g, nc, csz), headMajor(beta, nc, csz),
        consts[0],          consts[1],          consts[2],          consts[3],
    }, &.{
        f32s(.{ bh, nc, csz, vd }), f32s(.{ bh, nc, csz, kd }), f32s(.{ bh, nc, csz, kd }),  f32s(.{ bh, nc, csz, csz }),
        f32s(.{ bh, nc, kd, csz }), f32s(.{ bh, nc, 1, 1 }),    f32s(.{ bh, nc, csz, csz }),
    }, .{ .kernel_id = "zml.tt.chunk_gdn_prep", .artifact = prep_art });

    // tt-metal's state is `[b * h, k, v]`.
    const scan_art = scanProgram(allocator, @intCast(bh), @intCast(nc), opts.chunk_size, @intCast(kd), @intCast(vd), opts.grid) catch |err|
        std.debug.panic("gatedDeltaNet scan: {}", .{err});
    const scan = zml.ops.ttKernel(&.{ prep[0], prep[1], prep[2], prep[3], prep[4], prep[5], prep[6], s0.convert(.f32).reshape(.{ bh, kd, vd }) }, &.{
        f32s(.{ bh, nc, csz, vd }), f32s(.{ bh, kd, vd }),
    }, .{ .kernel_id = "zml.tt.chunk_gdn_scan", .artifact = scan_art });

    const out = scan[0].reshape(.{ .b = b, .h = h, .s = len, .v = vd }).slice(.s, .{ .end = s }).transpose(.{ .b, .s, .h, .v });
    const out_active = active.broad(out.shape().withDtype(.bool)).select(out, Tensor.scalar(0, .f32).broad(out.shape()));
    const state = scan[1].reshape(.{ .b = b, .h = h, .k = kd, .v = vd });
    const unlift = struct {
        fn f(t: Tensor, like: zml.Shape, batched: bool) Tensor {
            return (if (batched) t else t.squeeze(.b)).transpose(like).convert(like.dtype());
        }
    }.f;
    return .{
        .outputs = unlift(out_active, values.shape(), had_b),
        .state = .{ .s = unlift(state, initial_state.s.shape(), had_b) },
    };
}
