const std = @import("std");

const zml = @import("zml");

const log = std.log.scoped(.minimax_h3_sol_attn);

// =============================================================================
// refine/sol_attn.zig — SM100 Sol-Attn custom call
//
// Kernel is 32 heads. Gather .h to replicated, run, then re-shard .h=.model.
// =============================================================================

pub const heads: i64 = 32;
pub const head_dim: i64 = 128;
pub const min_tokens: i64 = 4096;
pub const block_size: i64 = 64;

pub fn tokensOk(tokens: i64) bool {
    return tokens >= min_tokens and @rem(tokens, block_size) == 0;
}
const log2e: f32 = 1.4426950408889634;

const Input = struct {
    q: zml.Tensor,
    k: zml.Tensor,
    v: zml.Tensor,
    kc: zml.Tensor,
    vc: zml.Tensor,
    thr: zml.Tensor,
};
const Output = struct { o: zml.Shape };
const Attrs = struct { scale: f32 };

const SolCall = zml.ops.CustomCall(Input, Output, Attrs, solFfi, .{
    .name = "h3$sol_attn",
    // Inputs are gathered to the full head count (any TP degree). Each replica
    // then sees the complete 32-head tensor.
    .sharding_aware = true,
    .has_side_effect = false,
});

const LaunchFn = *const fn (
    ?*const anyopaque,
    ?*const anyopaque,
    ?*const anyopaque,
    ?*anyopaque,
    ?*const anyopaque,
    ?*const anyopaque,
    ?*const anyopaque,
    ?*anyopaque,
    i32,
    i32,
    i32,
    i32,
    f32,
    ?*const anyopaque,
    i32,
) callconv(.c) c_int;

const Driver = struct {
    get_dev: *const fn (*c_int) callconv(.c) c_int,
    set_dev: *const fn (c_int) callconv(.c) c_int,
    malloc: *const fn (*?*anyopaque, usize) callconv(.c) c_int,
    free: *const fn (?*anyopaque) callconv(.c) c_int,
};

const Workspace = struct {
    ptr: ?*anyopaque = null,
    bytes: usize = 0,
};

/// LSE scratch is per CUDA ordinal after `CUDA_VISIBLE_DEVICES`. Sized for a
/// full mesh, not a 2-GPU box. Official H3 TP caps at head-gcd (8); extra GPUs
/// are dropped from the mesh before this runs.
const max_devices = 64;

var launch_fn: ?LaunchFn = null;
var driver_cache: ?Driver = null;
var lse_ws: [max_devices]Workspace = @splat(.{});
var load_lock: std.atomic.Value(bool) = .init(false);

pub fn register(platform: *const zml.Platform) !void {
    try SolCall.register(platform);
}

/// Official Super self-attn. `q,k,v` are FA2-tagged `.b .q/.k .h .hd` bf16.
/// Layer 0 uses `tau <= 0` so every block is exact; layers 1–47 use diag Sol-Attn.
///
/// The cute kernel is compiled for `heads` (32), not `heads / tp`. Any tensor-parallel
/// degree all-gathers `.h` first, then shards the output again for the rest of the block.
pub fn forward(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, tau: zml.Tensor) zml.Tensor {
    const q_t = q.withPartitioning(.{ .h = .replicated }).transpose(.{ .b, .q, .h, .hd });
    const k_t = k.withPartitioning(.{ .h = .replicated }).transpose(.{ .b, .k, .h, .hd });
    const v_t = v.withPartitioning(.{ .h = .replicated }).transpose(.{ .b, .k, .h, .hd });
    const tokens = q_t.dim(.q);
    const n_blk = @divExact(tokens, block_size);
    const qf = q_t.convert(.f32);
    const kf = k_t.convert(.f32);
    const vf = v_t.convert(.f32);
    const qb = qf.reshape(.{
        .b = qf.dim(.b),
        .nb = n_blk,
        .blk = block_size,
        .h = qf.dim(.h),
        .hd = head_dim,
    }).mean(.blk).squeeze(.blk);
    const kc = kf.reshape(.{
        .b = kf.dim(.b),
        .nb = n_blk,
        .blk = block_size,
        .h = kf.dim(.h),
        .hd = head_dim,
    }).mean(.blk).squeeze(.blk);
    const vc = vf.reshape(.{
        .b = vf.dim(.b),
        .nb = n_blk,
        .blk = block_size,
        .h = vf.dim(.h),
        .hd = head_dim,
    }).sum(.blk).squeeze(.blk);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    const log2_scale = scale * log2e;
    const kc_mean = kc.mean(.nb).squeeze(.nb);
    const kc_var = kc.mul(kc).mean(.nb).squeeze(.nb).sub(kc_mean.mul(kc_mean)).maximum(zml.Tensor.scalar(0, .f32));
    const mean = qb.mul(kc_mean.broad(qb.shape())).sum(.hd).squeeze(.hd).scale(log2_scale);
    const variance = qb.mul(qb).mul(kc_var.broad(qb.shape())).sum(.hd).squeeze(.hd).scale(log2_scale * log2_scale);
    const stddev = variance.add(zml.Tensor.scalar(1.0e-6, .f32)).sqrt();
    const sparse = mean.add(tau.convert(.f32).broad(mean.shape()).mul(stddev));
    const dense = tau.convert(.f32).cmp(.LE, zml.Tensor.scalar(0, .f32));
    const thr = dense.broad(sparse.shape()).select(
        zml.Tensor.scalar(-1.0e9, .f32).broad(sparse.shape()),
        sparse,
    );
    const o = SolCall.call(
        .{
            .q = q_t,
            .k = k_t,
            .v = v_t,
            .kc = kc.convert(.bf16),
            .vc = vc.convert(.bf16),
            .thr = thr,
        },
        .{ .o = q_t.shape() },
        .{ .scale = scale },
    ).o;
    return o.transpose(q.shape()).withPartitioning(.{ .h = .model });
}

fn solFfi(
    call_frame: *zml.pjrt.ffi.CallFrame,
    input: zml.pjrtx.TensorToCustomCallBuffer(Input),
    output: zml.pjrtx.ShapeToCustomCallBuffer(Output),
    attrs: Attrs,
) !?*zml.pjrt.ffi.Error {
    const stream = call_frame.stream() orelse return error.NoCudaStream;
    try launchOfficial(stream, input.q, input.k, input.v, input.kc, input.vc, input.thr, output.o, attrs.scale);
    return null;
}

fn lock() void {
    while (load_lock.cmpxchgWeak(false, true, .acquire, .monotonic) != null) {}
}

fn unlock() void {
    load_lock.store(false, .release);
}

fn loadSo(name: [:0]const u8) ?*anyopaque {
    return std.c.dlopen(name, .{ .LAZY = true, .GLOBAL = true });
}

fn req(so: ?*anyopaque, name: [:0]const u8) !*const anyopaque {
    if (std.c.dlsym(so, name)) |p| return p;
    if (std.c.dlsym(null, name)) |p| return p;
    log.err("missing symbol {s}", .{name});
    return error.CudaSymbol;
}

fn fnPtr(comptime T: type, p: *const anyopaque) T {
    return @ptrCast(@alignCast(p));
}

fn loadDriver() !Driver {
    if (driver_cache) |a| return a;
    const rt = loadSo("libcudart.so.13") orelse loadSo("libcudart.so");
    const a = Driver{
        .get_dev = fnPtr(@TypeOf(@as(Driver, undefined).get_dev), try req(rt, "cudaGetDevice")),
        .set_dev = fnPtr(@TypeOf(@as(Driver, undefined).set_dev), try req(rt, "cudaSetDevice")),
        .malloc = fnPtr(@TypeOf(@as(Driver, undefined).malloc), try req(rt, "cudaMalloc")),
        .free = fnPtr(@TypeOf(@as(Driver, undefined).free), try req(rt, "cudaFree")),
    };
    driver_cache = a;
    return a;
}

fn candidateLibs(buf: *[8][:0]const u8) []const [:0]const u8 {
    var n: usize = 0;
    if (std.c.getenv("SOL_ATTN_LIB")) |p| {
        buf[n] = std.mem.span(p);
        n += 1;
    }
    const extras = [_][:0]const u8{
        "output/sol-attn/libh3_sol_attn_sm100.so",
    };
    for (extras) |p| {
        buf[n] = p;
        n += 1;
    }
    return buf[0..n];
}

fn loadLaunch() !LaunchFn {
    if (launch_fn) |f| return f;
    var paths: [8][:0]const u8 = undefined;
    const cands = candidateLibs(&paths);
    var so: ?*anyopaque = null;
    for (cands) |p| {
        so = loadSo(p);
        if (so != null) {
            log.info("official cute_sm100 {s}", .{p});
            break;
        }
    }
    if (so == null) {
        log.err("libh3_sol_attn_sm100.so not found (set SOL_ATTN_LIB)", .{});
        return error.SolAttnLib;
    }
    const f = fnPtr(LaunchFn, try req(so, "h3_sol_attn_sm100"));
    launch_fn = f;
    return f;
}

fn lseBytes(tokens: u32) usize {
    return @as(usize, tokens) * @as(usize, @intCast(heads)) * @sizeOf(f32);
}

/// Allocate LSE scratch on every mesh GPU before the first Sol-Attn launch.
/// XLA command buffers capture this pointer; never free or move it afterward.
pub fn reserveWorkspace(max_tokens: u32, devices: u32) !void {
    if (max_tokens == 0 or devices == 0) return;
    if (devices > max_devices) {
        log.err("sol-attn workspace devices={d} exceeds {d}", .{ devices, max_devices });
        return error.SolAttnDevices;
    }
    lock();
    defer unlock();
    const drv = try loadDriver();
    const bytes = lseBytes(max_tokens);
    var d: u32 = 0;
    while (d < devices) : (d += 1) {
        _ = try lsePtr(drv, @intCast(d), bytes);
    }
    log.info("sol-attn workspace {d} tokens × {d} GPU", .{ max_tokens, devices });
}

fn lsePtr(drv: Driver, dev: c_int, bytes: usize) !?*anyopaque {
    if (dev < 0 or dev >= max_devices) return error.CudaGetDevice;
    const slot = &lse_ws[@intCast(dev)];
    if (slot.ptr != null and slot.bytes >= bytes) return slot.ptr;
    if (drv.set_dev(dev) != 0) return error.CudaSetDevice;
    var p: ?*anyopaque = null;
    if (drv.malloc(&p, bytes) != 0) return error.CudaMalloc;
    // Leave any smaller buffer alive. Command-buffer replay may still use it.
    slot.* = .{ .ptr = p, .bytes = bytes };
    return p;
}

fn launchOfficial(
    stream: *const anyopaque,
    q: zml.pjrtx.CustomCallBuffer,
    k: zml.pjrtx.CustomCallBuffer,
    v: zml.pjrtx.CustomCallBuffer,
    kc: zml.pjrtx.CustomCallBuffer,
    vc: zml.pjrtx.CustomCallBuffer,
    thr: zml.pjrtx.CustomCallBuffer,
    o: zml.pjrtx.CustomCallBuffer,
    scale: f32,
) !void {
    if (q.shape.dtype() != .bf16 or o.shape.dtype() != .bf16) return error.SolAttnDtype;
    if (q.shape.rank() != 4) return error.SolAttnRank;
    const b: i32 = @intCast(q.shape.dim(0));
    const t: i32 = @intCast(q.shape.dim(1));
    const h: i32 = @intCast(q.shape.dim(2));
    const d: i32 = @intCast(q.shape.dim(3));
    if (d != head_dim) return error.SolAttnHeadDim;
    if (h != heads) {
        log.err("sol-attn heads={d} (need {d} after all-gather, any TP degree)", .{ h, heads });
        return error.SolAttnHeads;
    }
    const n: i32 = @intCast(@divExact(t, block_size));

    const cached = blk: {
        lock();
        defer unlock();
        const drv = try loadDriver();
        const func = try loadLaunch();
        var dev: c_int = 0;
        if (drv.get_dev(&dev) != 0) return error.CudaGetDevice;
        const bytes: usize = @intCast(b * t * h * @sizeOf(f32));
        const lse = try lsePtr(drv, dev, bytes);
        break :blk .{ .func = func, .dev = dev, .lse = lse };
    };

    const st = cached.func(
        q.ptr,
        k.ptr,
        v.ptr,
        o.ptr,
        kc.ptr,
        vc.ptr,
        thr.ptr,
        cached.lse,
        b,
        t,
        h,
        n,
        scale,
        stream,
        cached.dev,
    );
    if (st != 0) {
        log.err("h3_sol_attn_sm100: {d}", .{st});
        return error.SolAttnLaunch;
    }
}
