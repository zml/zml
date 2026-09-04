const std = @import("std");

const zml = @import("zml");

const sku = @import("../recipe/sku.zig");
const weights = @import("../recipe/weights.zig");

const log = std.log.scoped(.minimax_h3_stage2);

// =============================================================================
// refine/taehv.zig — TAEHV decode + HD chunk stitch
//
// HD 10s/15s decode t=16 windows. drop = written - start*8, not overlap*8.
// =============================================================================

pub const default_path = "output/taeltx2_3_wide.safetensors";
pub const weight_paths = [_][]const u8{
    default_path,
    "/var/models/super-accel/ltx/taehv/taeltx2_3_wide.safetensors",
    sku.http_taehv,
};
/// TAEHV expands time ×8 then drops this many leading frames (`t*8 - 7`).
pub const temporal_expand: u32 = 8;
pub const temporal_trim: i64 = 7;

pub fn outFrames(latent_t: u32) u32 {
    return latent_t * temporal_expand - @as(u32, @intCast(temporal_trim));
}

/// Local frames to skip so chunk `latent_start` continues a stream that already
/// has `written` pixel frames. Each latent maps to `temporal_expand` raw frames;
/// the decoder trims `temporal_trim` from every window independently.
pub fn chunkDrop(written: u32, latent_start: u32) u32 {
    const origin = latent_start * temporal_expand;
    return if (written > origin) written - origin else 0;
}

/// Pixel frames the host stitch can emit. Must equal `outFrames(t_all)` for
/// `chunk_t`/`overlap` used at runtime (HD 10s/15s: 16/8).
pub fn chunkedCoverage(chunk_t: u32, overlap: u32, t_all: u32) u32 {
    if (overlap >= chunk_t or t_all == 0) return 0;
    const keep_t = outFrames(t_all);
    const step = chunk_t - overlap;
    const chunk_out = outFrames(chunk_t);
    var written: u32 = 0;
    var start: u32 = 0;
    while (written < keep_t) {
        const drop = chunkDrop(written, start);
        var f = drop;
        while (f < chunk_out and written < keep_t) : (f += 1) written += 1;
        if (f == drop) break;
        if (start + chunk_t >= t_all) break;
        start += step;
    }
    return written;
}

const Conv = struct {
    weight: zml.Tensor,
    bias: ?zml.Tensor = null,
};

pub const WideMem = struct {
    c0: Conv,
    c2: Conv,
    c4: Conv,
    c6: Conv,
};

pub const Model = struct {
    batch: i64,
    time: i64,
    stem: Conv,
    m0: [3]WideMem,
    grow0: Conv,
    down0: Conv,
    m1: [3]WideMem,
    grow1: Conv,
    down1: Conv,
    m2: [3]WideMem,
    grow2: Conv,
    down2: Conv,
    head: Conv,

    pub fn init(store: zml.io.TensorStore.View, batch: i64, time: i64) Model {
        const d = store.withPrefix("decoder");
        return .{
            .batch = batch,
            .time = time,
            .stem = conv(d, "1", true),
            .m0 = .{ mem(d, "3"), mem(d, "4"), mem(d, "5") },
            .grow0 = conv1(d, "7.conv"),
            .down0 = conv(d, "8", false),
            .m1 = .{ mem(d, "9"), mem(d, "10"), mem(d, "11") },
            .grow1 = conv1(d, "13.conv"),
            .down1 = conv(d, "14", false),
            .m2 = .{ mem(d, "15"), mem(d, "16"), mem(d, "17") },
            .grow2 = conv1(d, "19.conv"),
            .down2 = conv(d, "20", false),
            .head = conv(d, "22", true),
        };
    }
};

fn conv(store: zml.io.TensorStore.View, id: []const u8, bias: bool) Conv {
    var wname: [32]u8 = undefined;
    var bname: [32]u8 = undefined;
    const w = std.fmt.bufPrint(&wname, "{s}.weight", .{id}) catch unreachable;
    const b = std.fmt.bufPrint(&bname, "{s}.bias", .{id}) catch unreachable;
    return .{
        .weight = store.createTensor(w, .{ .co, .ci, .kh, .kw }, .replicated),
        .bias = if (bias) store.createTensor(b, .{.co}, .replicated) else null,
    };
}

fn conv1(store: zml.io.TensorStore.View, id: []const u8) Conv {
    var wname: [40]u8 = undefined;
    const w = std.fmt.bufPrint(&wname, "{s}.weight", .{id}) catch unreachable;
    return .{
        .weight = store.createTensor(w, .{ .co, .ci, .kh, .kw }, .replicated),
        .bias = null,
    };
}

fn convNamed(store: zml.io.TensorStore.View, id: []const u8, bias: bool) Conv {
    var wname: [48]u8 = undefined;
    var bname: [48]u8 = undefined;
    const w = std.fmt.bufPrint(&wname, "{s}.weight", .{id}) catch unreachable;
    const b = std.fmt.bufPrint(&bname, "{s}.bias", .{id}) catch unreachable;
    return .{
        .weight = store.createTensor(w, .{ .co, .ci, .kh, .kw }, .replicated),
        .bias = if (bias) store.createTensor(b, .{.co}, .replicated) else null,
    };
}

fn mem(store: zml.io.TensorStore.View, id: []const u8) WideMem {
    var p0: [40]u8 = undefined;
    var p2: [40]u8 = undefined;
    var p4: [40]u8 = undefined;
    var p6: [40]u8 = undefined;
    return .{
        .c0 = convNamed(store, std.fmt.bufPrint(&p0, "{s}.conv.0", .{id}) catch unreachable, true),
        .c2 = convNamed(store, std.fmt.bufPrint(&p2, "{s}.conv.2", .{id}) catch unreachable, true),
        .c4 = convNamed(store, std.fmt.bufPrint(&p4, "{s}.conv.4", .{id}) catch unreachable, true),
        .c6 = convNamed(store, std.fmt.bufPrint(&p6, "{s}.conv.6", .{id}) catch unreachable, true),
    };
}

fn asNchw(x: zml.Tensor) zml.Tensor {
    return x.withTags(.{ .n, .c, .h, .w });
}

fn asDt(x: zml.Tensor, t: zml.Tensor) zml.Tensor {
    return if (x.dtype() == t.dtype()) x else x.convert(t.dtype());
}

fn groupsFor(c: Conv, in_c: i64) i64 {
    const ci = c.weight.dim(.ci);
    if (ci == in_c) return 1;
    if (@rem(in_c, ci) == 0) return @divExact(in_c, ci);
    return 1;
}

fn applyConv(x: zml.Tensor, c: Conv) zml.Tensor {
    const groups = groupsFor(c, x.dim(.c));
    var y = if (groups == 1)
        asNchw(x.conv2d(asDt(c.weight, x), .{ .padding = &.{ 1, 1, 1, 1 } }))
    else
        groupedSpatialConv(x, c, groups);
    if (c.bias) |b| y = y.add(asDt(b, y).rename(.{ .co = .c }).broad(y.shape()));
    return y;
}

fn groupedSpatialConv(x: zml.Tensor, c: Conv, groups: i64) zml.Tensor {
    const in_g = @divExact(x.dim(.c), groups);
    const out_g = @divExact(c.weight.dim(.co), groups);
    const w = asDt(c.weight, x);
    var parts: [16]zml.Tensor = undefined;
    const n_g: usize = @intCast(groups);
    var gi: i64 = 0;
    while (gi < groups) : (gi += 1) {
        const xin = x.slice(.c, .{ .start = gi * in_g, .end = (gi + 1) * in_g });
        const wg = w.slice(.co, .{ .start = gi * out_g, .end = (gi + 1) * out_g });
        parts[@intCast(gi)] = asNchw(xin.conv2d(wg, .{ .padding = &.{ 1, 1, 1, 1 } }));
    }
    return zml.Tensor.concatenate(parts[0..n_g], .c);
}

fn applyConv1(x: zml.Tensor, c: Conv) zml.Tensor {
    var y = asNchw(x.conv2d(asDt(c.weight, x), .{}));
    if (c.bias) |b| y = y.add(asDt(b, y).rename(.{ .co = .c }).broad(y.shape()));
    return y;
}

fn memblock(m: WideMem, x: zml.Tensor, batch: i64, time: i64) zml.Tensor {
    const xt = x.reshape(.{ .b = batch, .t = time, .c = x.dim(.c), .h = x.dim(.h), .w = x.dim(.w) });
    const past = xt.pad(0, .{ .t = zml.Tensor.Pad{ .low = 1 } }).slice(.t, .{ .start = 0, .end = time });
    const p4 = past.reshape(.{ .n = x.dim(.n), .c = x.dim(.c), .h = x.dim(.h), .w = x.dim(.w) });
    const cat = zml.Tensor.concatenate(&.{ x, p4 }, .c);
    var y = applyConv1(cat, m.c0).relu();
    y = applyConv(y, m.c2).relu();
    y = applyConv1(y, m.c4).relu();
    y = applyConv(y, m.c6);
    return y.add(x).relu();
}

fn tgrow(c: Conv, x: zml.Tensor, stride: i64) zml.Tensor {
    const y = applyConv1(x, c);
    if (stride == 1) return y;
    return y.reshape(.{
        .n = y.dim(.n) * stride,
        .c = @divExact(y.dim(.c), stride),
        .h = y.dim(.h),
        .w = y.dim(.w),
    });
}

fn up2(x: zml.Tensor) zml.Tensor {
    return asNchw(zml.nn.upsample(x, .{ .mode = .nearest, .scale_factor = &.{ 2, 2 } }));
}

fn pixelShuffle4(x: zml.Tensor) zml.Tensor {
    const n = x.dim(.n);
    const h = x.dim(.h);
    const w = x.dim(.w);
    const y = x.reshape(.{ .n = n, .c = 3, .rh = 4, .rw = 4, .h = h, .w = w });
    const z = y.transpose(.{ .n, .c, .h, .rh, .w, .rw });
    return asNchw(z.reshape(.{ .n = n, .c = 3, .h = h * 4, .w = w * 4 }));
}

pub const Input = struct {
    model: Model,
    latent: zml.Tensor,
};

pub const Output = struct {
    rgb: zml.Tensor,
};

pub fn decode(input: Input) Output {
    const model = input.model;
    const b = model.batch;
    var t = model.time;
    // DiT emits NCTHW. Fold N×T into the conv batch inside this graph so
    // hot refine never D2H/H2D the latent just to change axis order.
    const c = input.latent.dim(.c);
    const h = input.latent.dim(.h);
    const w = input.latent.dim(.w);
    var x = input.latent.transpose(.{ .n, .t, .c, .h, .w }).reshape(.{
        .n = b * t,
        .c = c,
        .h = h,
        .w = w,
    });
    x = x.scale(1.0 / 3.0).tanh().scale(3.0);
    x = applyConv(x, model.stem).relu();
    x = memblock(model.m0[0], x, b, t);
    x = memblock(model.m0[1], x, b, t);
    x = memblock(model.m0[2], x, b, t);
    x = up2(x);
    x = tgrow(model.grow0, x, 2);
    t *= 2;
    x = applyConv(x, model.down0);
    for (model.m1) |m| x = memblock(m, x, b, t);
    x = up2(x);
    x = tgrow(model.grow1, x, 2);
    t *= 2;
    x = applyConv(x, model.down1);
    for (model.m2) |m| x = memblock(m, x, b, t);
    x = up2(x);
    x = tgrow(model.grow2, x, 2);
    x = applyConv(x, model.down2).relu();
    x = applyConv(x, model.head);
    x = pixelShuffle4(x);
    const zero = zml.Tensor.scalar(0.0, x.dtype());
    const one = zml.Tensor.scalar(1.0, x.dtype());
    x = x.maximum(zero).minimum(one);
    // Drop the 7-frame TAEHV warmup and emit C,T,H,W so remux D2H is one
    // contiguous gather — no host plane shuffle of 1.5 GiB.
    x = x.slice(.n, .{ .start = temporal_trim, .end = x.dim(.n) });
    return .{ .rgb = x.transpose(.{ .c, .n, .h, .w }) };
}

pub const Compiled = struct {
    decode: zml.FnExe(decode),
    bufs: zml.Bufferized(Model),
    batch: u32,
    time: u32,
    latent_h: u32,
    latent_w: u32,
    owns_bufs: bool = true,

    pub fn deinit(self: *Compiled) void {
        if (self.owns_bufs) zml.Buffer.deinitAll(Model, &self.bufs);
        self.decode.deinit();
    }

    pub fn matches(self: *const Compiled, time: i64, h: u32, w: u32) bool {
        return self.time == @as(u32, @intCast(time)) and self.latent_h == h and self.latent_w == w;
    }
};

pub fn compile(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model: Model,
    latent_h: u32,
    latent_w: u32,
    shardings: []const zml.Sharding,
    store: *zml.io.TensorStore,
    progress: *std.Progress.Node,
    reuse: ?*const Compiled,
) !Compiled {
    progress.increaseEstimatedTotalItems(1);
    const exe = try zml.FnExe(decode).compile(allocator, io, platform, .{
        .shardings = shardings,
        .program_name = "minimax_h3_taehv",
    }, .{.{
        .model = model,
        .latent = .init(.{ .n = 1, .c = 128, .t = model.time, .h = latent_h, .w = latent_w }, .f32),
    }});
    const bufs = if (reuse) |src| src.bufs else try weights.load(allocator, io, platform, store, shardings, Model, &model, progress, null);
    log.info("compile TAEHV Wide {d}x{d} t={d}{s}", .{
        latent_w,
        latent_h,
        model.time,
        if (reuse != null) " reuse weights" else "",
    });
    return .{
        .decode = exe,
        .bufs = bufs,
        .batch = @intCast(model.batch),
        .time = @intCast(model.time),
        .latent_h = latent_h,
        .latent_w = latent_w,
        .owns_bufs = reuse == null,
    };
}

pub const chunk_overlap: u32 = 8;
pub const chunk_blend: u32 = 8;
/// HD 10s/15s decode this many latent frames per window.
pub const window_t: i64 = 16;

/// Chunk when spatial height is at least `chunk_min_h` and time exceeds `window_t`.
pub fn decodeTime(latent_t: i64, latent_h: u32, chunk_min_h: u32) i64 {
    return if (latent_h >= chunk_min_h and latent_t > window_t) window_t else latent_t;
}

pub fn decodeLatentWith(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    exe: *Compiled,
    tae_run: *zml.FnExe(decode).Runner(.{.model}),
    latent: *zml.Buffer,
    time: u32,
    height: u32,
    width: u32,
) ![]f32 {
    const out_h = height * 32;
    const out_w = width * 32;
    const keep_t = outFrames(time);
    if (exe.time == time) {
        var rgb_b: zml.Buffer = undefined;
        tae_run.run(io, .{
            .inputs = .{ .latent = latent.* },
            .outputs = .{ .rgb = &rgb_b },
            .opts = .{ .wait = true },
        });
        defer rgb_b.deinit();
        latent.deinit();
        const nchw = try allocator.alloc(f32, 3 * keep_t * out_h * out_w);
        errdefer allocator.free(nchw);
        try rgb_b.toSlice(io, .init(rgb_b.shape(), std.mem.sliceAsBytes(nchw)));
        return nchw;
    }

    const chunk_t = exe.time;
    const overlap: u32 = @min(chunk_overlap, chunk_t / 2);
    const step = chunk_t - overlap;
    const c: u32 = 128;
    const sl = try latent.toSliceAlloc(allocator, io);
    defer sl.free(allocator);
    latent.deinit();
    const src = @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, sl.data())));

    const nchw = try allocator.alloc(f32, 3 * keep_t * out_h * out_w);
    errdefer allocator.free(nchw);
    const chunk_host = try allocator.alloc(f32, c * chunk_t * height * width);
    defer allocator.free(chunk_host);
    const chunk_out_t = outFrames(chunk_t);
    const rgb_host = try allocator.alloc(f32, 3 * chunk_out_t * out_h * out_w);
    defer allocator.free(rgb_host);

    var written: u32 = 0;
    var start: u32 = 0;
    while (written < keep_t) {
        copyLatentChunk(chunk_host, src, start, chunk_t, time, c, height, width);
        var chunk_b = try weights.fromItems(io, platform, .init(.{ .n = 1, .c = 128, .t = chunk_t, .h = height, .w = width }, .f32), chunk_host);
        defer chunk_b.deinit();
        var rgb_b: zml.Buffer = undefined;
        tae_run.run(io, .{
            .inputs = .{ .latent = chunk_b },
            .outputs = .{ .rgb = &rgb_b },
            .opts = .{ .wait = true },
        });
        defer rgb_b.deinit();
        try rgb_b.toSlice(io, .init(rgb_b.shape(), std.mem.sliceAsBytes(rgb_host)));
        appendChunk(nchw, rgb_host, &written, start, keep_t, chunk_out_t, out_h * out_w, chunk_blend);
        if (start + chunk_t >= time) break;
        start += step;
    }
    if (written != keep_t) return error.TaehvChunk;
    log.info("TAEHV chunked t={d} overlap={d} -> {d} frames", .{ time, overlap, keep_t });
    return nchw;
}

/// Copy one TAEHV window into `dst`. `chunk[f]` is treated as global frame `start*8 + f`.
pub fn appendChunk(
    dst: []f32,
    chunk: []const f32,
    written: *u32,
    start: u32,
    keep_t: u32,
    chunk_out_t: u32,
    plane: u32,
    blend: u32,
) void {
    const drop = chunkDrop(written.*, start);
    if (drop > 0 and written.* > 0) {
        const b = @min(blend, @min(drop, written.*));
        var bi: u32 = 0;
        while (bi < b) : (bi += 1) {
            const alpha = @as(f32, @floatFromInt(bi + 1)) / @as(f32, @floatFromInt(b + 1));
            const src_f = drop - b + bi;
            const dst_f = written.* - b + bi;
            var ch: u32 = 0;
            while (ch < 3) : (ch += 1) {
                const src_i = (ch * chunk_out_t + src_f) * plane;
                const dst_i = (ch * keep_t + dst_f) * plane;
                lerpPlane(dst[dst_i..][0..plane], chunk[src_i..][0..plane], alpha);
            }
        }
    }
    var f: u32 = drop;
    while (f < chunk_out_t and written.* < keep_t) : (f += 1) {
        var ch: u32 = 0;
        while (ch < 3) : (ch += 1) {
            const src_i = (ch * chunk_out_t + f) * plane;
            const dst_i = (ch * keep_t + written.*) * plane;
            @memcpy(dst[dst_i..][0..plane], chunk[src_i..][0..plane]);
        }
        written.* += 1;
    }
}

pub fn stitchPattern(dst: []f32, chunk_t: u32, overlap: u32, t_all: u32, blend: u32) u32 {
    const keep_t = outFrames(t_all);
    const step = chunk_t - overlap;
    const chunk_out_t = outFrames(chunk_t);
    var chunk: [3 * 256]f32 = undefined;
    var written: u32 = 0;
    var start: u32 = 0;
    while (written < keep_t) {
        var f: u32 = 0;
        while (f < chunk_out_t) : (f += 1) {
            const v = @as(f32, @floatFromInt(start * temporal_expand + f));
            chunk[0 * chunk_out_t + f] = v;
            chunk[1 * chunk_out_t + f] = v;
            chunk[2 * chunk_out_t + f] = v;
        }
        appendChunk(dst[0 .. 3 * keep_t], chunk[0 .. 3 * chunk_out_t], &written, start, keep_t, chunk_out_t, 1, blend);
        if (start + chunk_t >= t_all) break;
        start += step;
    }
    return written;
}

fn lerpPlane(dst: []f32, src: []const f32, alpha: f32) void {
    const keep = 1.0 - alpha;
    for (dst, src) |*d, s| d.* = d.* * keep + s * alpha;
}

fn copyLatentChunk(dst: []f32, src: []const f32, start: u32, chunk_t: u32, src_t: u32, c: u32, h: u32, w: u32) void {
    const hw = h * w;
    var cc: u32 = 0;
    while (cc < c) : (cc += 1) {
        var tt: u32 = 0;
        while (tt < chunk_t) : (tt += 1) {
            const src_tt = @min(start + tt, src_t - 1);
            const src_off = (cc * src_t + src_tt) * hw;
            const dst_off = (cc * chunk_t + tt) * hw;
            @memcpy(dst[dst_off..][0..hw], src[src_off..][0..hw]);
        }
    }
}
