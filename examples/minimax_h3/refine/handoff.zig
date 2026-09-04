const std = @import("std");

const zml = @import("zml");

const sku = @import("../recipe/sku.zig");
const taeh3 = @import("../draft/taeh3.zig");
const weights = @import("../recipe/weights.zig");

const log = std.log.scoped(.minimax_h3);

// =============================================================================
// refine/handoff.zig — TAEH3 RGB → LTX VAE pixels
//
// Scale-to-cover, center-crop, bicubic Keys-AA. Fail = HandoffMismatch.
// =============================================================================

/// Official `prepare_h3_draft_for_ltx_refiner`: scale-to-cover, center-crop,
/// `F.interpolate(..., mode="bicubic", align_corners=False, antialias=True)`.
pub const Prepared = struct {
    rgb: []f32,
    frames: u32,
    width: u32,
    height: u32,

    pub fn deinit(self: Prepared, allocator: std.mem.Allocator) void {
        allocator.free(self.rgb);
    }
};

pub fn trimFrames(frames: u32) u32 {
    if (frames < 1) return 1;
    return 1 + ((frames - 1) / 8) * 8;
}

pub const Cover = struct {
    resized_h: u32,
    resized_w: u32,
    out_h: u32,
    out_w: u32,
    top: u32,
    left: u32,

    pub fn init(src_h: u32, src_w: u32, out_h: u32, out_w: u32) Cover {
        const scale = @max(
            @as(f32, @floatFromInt(out_h)) / @as(f32, @floatFromInt(src_h)),
            @as(f32, @floatFromInt(out_w)) / @as(f32, @floatFromInt(src_w)),
        );
        const resized_h: u32 = @max(out_h, @as(u32, @intFromFloat(@ceil(@as(f32, @floatFromInt(src_h)) * scale))));
        const resized_w: u32 = @max(out_w, @as(u32, @intFromFloat(@ceil(@as(f32, @floatFromInt(src_w)) * scale))));
        return .{
            .resized_h = resized_h,
            .resized_w = resized_w,
            .out_h = out_h,
            .out_w = out_w,
            .top = (resized_h - out_h) / 2,
            .left = (resized_w - out_w) / 2,
        };
    }
};

/// `nchw` is official TAEH3 layout C,T,H,W.
pub fn prepareDraft(
    allocator: std.mem.Allocator,
    nchw: []const f32,
    src_t: u32,
    src_h: u32,
    src_w: u32,
    target_w: u32,
    target_h: u32,
) !Prepared {
    const enc = try sku.refineEncodeSize(target_w, target_h);
    const kept = trimFrames(src_t);
    const rgb = try resizeThenCrop(allocator, nchw, src_t, kept, src_h, src_w, enc.h, enc.w);
    return .{
        .rgb = rgb,
        .frames = kept,
        .width = enc.w,
        .height = enc.h,
    };
}

pub fn resizeThenCrop(
    allocator: std.mem.Allocator,
    nchw: []const f32,
    src_t: u32,
    frames: u32,
    src_h: u32,
    src_w: u32,
    out_h: u32,
    out_w: u32,
) ![]f32 {
    const cover = Cover.init(src_h, src_w, out_h, out_w);

    const src_nhwc = try allocator.alloc(f32, frames * src_h * src_w * 3);
    defer allocator.free(src_nhwc);
    var f: u32 = 0;
    while (f < frames) : (f += 1) {
        frameToNhwc(nchw, src_t, src_h, src_w, f, src_nhwc[f * src_h * src_w * 3 ..][0 .. src_h * src_w * 3]);
    }

    const resized = try allocator.alloc(f32, frames * cover.resized_h * cover.resized_w * 3);
    defer allocator.free(resized);
    var ff: u32 = 0;
    while (ff < frames) : (ff += 1) {
        try resizeBicubicAa(
            allocator,
            src_nhwc[ff * src_h * src_w * 3 ..][0 .. src_h * src_w * 3],
            src_w,
            src_h,
            resized[ff * cover.resized_h * cover.resized_w * 3 ..][0 .. cover.resized_h * cover.resized_w * 3],
            cover.resized_w,
            cover.resized_h,
        );
    }

    const out = try allocator.alloc(f32, frames * out_h * out_w * 3);
    var fi: u32 = 0;
    while (fi < frames) : (fi += 1) {
        var y: u32 = 0;
        while (y < out_h) : (y += 1) {
            var x: u32 = 0;
            while (x < out_w) : (x += 1) {
                const s = ((fi * cover.resized_h + (cover.top + y)) * cover.resized_w + (cover.left + x)) * 3;
                const d = ((fi * out_h + y) * out_w + x) * 3;
                out[d + 0] = resized[s + 0];
                out[d + 1] = resized[s + 1];
                out[d + 2] = resized[s + 2];
            }
        }
    }
    return out;
}

fn frameToNhwc(nchw: []const f32, src_t: u32, src_h: u32, src_w: u32, f: u32, dst: []f32) void {
    var y: u32 = 0;
    while (y < src_h) : (y += 1) {
        var x: u32 = 0;
        while (x < src_w) : (x += 1) {
            const d = (y * src_w + x) * 3;
            dst[d + 0] = nchw[((0 * src_t + f) * src_h + y) * src_w + x];
            dst[d + 1] = nchw[((1 * src_t + f) * src_h + y) * src_w + x];
            dst[d + 2] = nchw[((2 * src_t + f) * src_h + y) * src_w + x];
        }
    }
}

/// Torch `_upsample_bicubic2d_aa` Keys cubic (`a = -0.5`).
pub fn bicubicKeysAa(x: f64) f64 {
    const a: f64 = -0.5;
    const ax = @abs(x);
    if (ax < 1.0) return ((a + 2.0) * ax - (a + 3.0)) * ax * ax + 1.0;
    if (ax < 2.0) return ((a * ax - 5.0 * a) * ax + 8.0 * a) * ax - 4.0 * a;
    return 0;
}

const AaWindow = struct {
    xmin: i64,
    xsize: i64,
    scale: f64,
    support: f64,
    invscale: f64,
};

fn aaWindow(src_len: u32, dst_len: u32) AaWindow {
    const scale = @as(f64, @floatFromInt(src_len)) / @as(f64, @floatFromInt(dst_len));
    const support = if (scale >= 1.0) 2.0 * scale else 2.0;
    const invscale = if (scale >= 1.0) 1.0 / scale else 1.0;
    return .{
        .xmin = 0,
        .xsize = 0,
        .scale = scale,
        .support = support,
        .invscale = invscale,
    };
}

/// One dest row of the separable Keys AA kernel. `weights[0..xsize]` are already
/// divided by `wsum`, matching `resize1dAa`.
pub fn aaRow(src_len: u32, dst_len: u32, dest_i: u32, row: []f64) struct { xmin: i64, xsize: usize } {
    const win = aaWindow(src_len, dst_len);
    const max_interp: i64 = @as(i64, @intFromFloat(@ceil(win.support))) * 2 + 1;
    const i: f64 = @floatFromInt(dest_i);
    const center = win.scale * (i + 0.5);
    const xmin = @max(@as(i64, @intFromFloat(center - win.support + 0.5)), 0);
    var xsize = @min(@as(i64, @intFromFloat(center + win.support + 0.5)), @as(i64, @intCast(src_len))) - xmin;
    if (xsize < 0) xsize = 0;
    if (xsize > max_interp) xsize = max_interp;
    const n: usize = @intCast(xsize);
    std.debug.assert(row.len >= n);
    var wsum: f64 = 0;
    var j: usize = 0;
    while (j < n) : (j += 1) {
        const w = bicubicKeysAa((@as(f64, @floatFromInt(j + @as(usize, @intCast(xmin)))) - center + 0.5) * win.invscale);
        row[j] = w;
        wsum += w;
    }
    if (wsum == 0) wsum = 1;
    j = 0;
    while (j < n) : (j += 1) row[j] /= wsum;
    return .{ .xmin = xmin, .xsize = n };
}

/// Dense `[dst_len, src_len]` Keys AA matrix used by the device handoff.
pub fn fillAa1d(dst: []f32, src_len: u32, dst_len: u32) void {
    std.debug.assert(dst.len == @as(usize, dst_len) * src_len);
    @memset(dst, 0);
    if (src_len == dst_len) {
        var i: u32 = 0;
        while (i < dst_len) : (i += 1) dst[@as(usize, i) * src_len + i] = 1;
        return;
    }
    var row: [64]f64 = undefined;
    var i: u32 = 0;
    while (i < dst_len) : (i += 1) {
        const got = aaRow(src_len, dst_len, i, &row);
        var j: usize = 0;
        while (j < got.xsize) : (j += 1) {
            dst[@as(usize, i) * src_len + @as(usize, @intCast(got.xmin)) + j] = @floatCast(row[j]);
        }
    }
}

/// Host matmul of `fillAa1d` weights. Same algebra as the device handoff.
pub fn applyAa1d(
    src: []const f32,
    src_w: u32,
    src_h: u32,
    dst: []f32,
    dst_w: u32,
    dst_h: u32,
    horizontal: bool,
    matrix: []const f32,
) void {
    const src_len: u32 = if (horizontal) src_w else src_h;
    const dst_len: u32 = if (horizontal) dst_w else dst_h;
    std.debug.assert(matrix.len == @as(usize, dst_len) * src_len);
    var dy: u32 = 0;
    while (dy < dst_h) : (dy += 1) {
        var dx: u32 = 0;
        while (dx < dst_w) : (dx += 1) {
            const dest_i: u32 = if (horizontal) dx else dy;
            var acc = [3]f64{ 0, 0, 0 };
            var s: u32 = 0;
            while (s < src_len) : (s += 1) {
                const w = matrix[@as(usize, dest_i) * src_len + s];
                if (w == 0) continue;
                const sx: u32 = if (horizontal) s else dx;
                const sy: u32 = if (horizontal) dy else s;
                const si = (@as(usize, sy) * src_w + sx) * 3;
                inline for (0..3) |c| acc[c] += @as(f64, w) * src[si + c];
            }
            const di = (@as(usize, dy) * dst_w + dx) * 3;
            inline for (0..3) |c| dst[di + c] = @floatCast(acc[c]);
        }
    }
}

pub fn resizeBicubicAa(
    allocator: std.mem.Allocator,
    src: []const f32,
    src_w: u32,
    src_h: u32,
    dst: []f32,
    dst_w: u32,
    dst_h: u32,
) !void {
    if (src_w == dst_w and src_h == dst_h) {
        @memcpy(dst, src);
        return;
    }
    const mid = try allocator.alloc(f32, src_h * dst_w * 3);
    defer allocator.free(mid);
    resize1dAa(src, src_w, src_h, mid, dst_w, src_h, true);
    resize1dAa(mid, dst_w, src_h, dst, dst_w, dst_h, false);
}

fn resize1dAa(
    src: []const f32,
    src_w: u32,
    src_h: u32,
    dst: []f32,
    dst_w: u32,
    dst_h: u32,
    horizontal: bool,
) void {
    const src_len: u32 = if (horizontal) src_w else src_h;
    const dst_len: u32 = if (horizontal) dst_w else dst_h;
    var row: [64]f64 = undefined;
    var dy: u32 = 0;
    while (dy < dst_h) : (dy += 1) {
        var dx: u32 = 0;
        while (dx < dst_w) : (dx += 1) {
            const dest_i: u32 = if (horizontal) dx else dy;
            const got = aaRow(src_len, dst_len, dest_i, &row);
            var acc = [3]f64{ 0, 0, 0 };
            var j: usize = 0;
            while (j < got.xsize) : (j += 1) {
                const src_pos: u32 = @intCast(got.xmin + @as(i64, @intCast(j)));
                const sx: u32 = if (horizontal) src_pos else dx;
                const sy: u32 = if (horizontal) dy else src_pos;
                const si = (@as(usize, sy) * src_w + sx) * 3;
                inline for (0..3) |c| acc[c] += row[j] * src[si + c];
            }
            const di = (@as(usize, dy) * dst_w + dx) * 3;
            inline for (0..3) |c| dst[di + c] = @floatCast(acc[c]);
        }
    }
}

pub const Model = struct {
    idx: zml.Tensor,
    wh: zml.Tensor,
    wv: zml.Tensor,
    crop_top: u32,
    crop_left: u32,
    out_h: u32,
    out_w: u32,
};

pub const HandoffIn = struct {
    model: Model,
    rgb: zml.Tensor,
};

pub const HandoffOut = struct {
    pixels: zml.Tensor,
};

/// TAEH3 RGB `{n=t,c,h,w}` → official takeFrames + clamp + Keys AA → VAE NTHWC.
/// After the height `dot`, leftover axes are `{t,c,w,h}`; transpose back to `{t,c,h,w}`
/// before crop so reshape cannot relabel the wrong memory order.
pub fn handoff(input: HandoffIn) HandoffOut {
    const model = input.model;
    var x = input.rgb.withTags(.{ .n, .c, .h, .w });
    x = x.gather(.{ .n = model.idx.withTags(.{.t}) }, .{}).withTags(.{ .t, .c, .h, .w });
    x = x.clamp(zml.Tensor.scalar(0, x.dtype()), zml.Tensor.scalar(1, x.dtype()));
    const wh = model.wh.withTags(.{ .wo, .w });
    const wv = model.wv.withTags(.{ .ho, .h });
    x = x.dot(wh, .w).rename(.{ .wo = .w });
    x = x.dot(wv, .h).rename(.{ .ho = .h });
    x = x.transpose(.{ .t, .c, .h, .w });
    if (model.crop_top != 0 or model.out_h != x.dim(.h)) {
        x = x.slice(.h, .{ .start = model.crop_top, .end = model.crop_top + model.out_h });
    }
    if (model.crop_left != 0 or model.out_w != x.dim(.w)) {
        x = x.slice(.w, .{ .start = model.crop_left, .end = model.crop_left + model.out_w });
    }
    x = x.reshape(.{ .n = 1, .t = x.dim(.t), .c = x.dim(.c), .h = x.dim(.h), .w = x.dim(.w) });
    return .{ .pixels = x.transpose(.{ .n, .t, .h, .w, .c }) };
}

pub const Compiled = struct {
    forward: zml.FnExe(handoff),
    bufs: zml.Bufferized(Model),
    frames: u32,
    height: u32,
    width: u32,
    matches_cpu: bool = false,

    pub fn deinit(self: *Compiled) void {
        self.bufs.idx.deinit();
        self.bufs.wh.deinit();
        self.bufs.wv.deinit();
        self.forward.deinit();
    }

    pub fn run(self: *const Compiled, allocator: std.mem.Allocator, io: std.Io, rgb: zml.Buffer) !zml.Buffer {
        var runner = try zml.FnExe(handoff).Runner(.{.model}).init(&self.forward, allocator, .{ .model = self.bufs });
        defer runner.deinit(allocator);
        var pixels: zml.Buffer = undefined;
        runner.run(io, .{
            .inputs = .{ .rgb = rgb },
            .outputs = .{ .pixels = &pixels },
            .opts = .{ .wait = true },
        });
        return pixels;
    }
};

pub fn compile(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shardings: []const zml.Sharding,
    progress: *std.Progress.Node,
    src_t: u32,
    src_h: u32,
    src_w: u32,
    src_frames: u32,
    target_w: u32,
    target_h: u32,
) !*Compiled {
    const out = try allocator.create(Compiled);
    errdefer allocator.destroy(out);
    progress.increaseEstimatedTotalItems(1);
    const enc = try sku.refineEncodeSize(target_w, target_h);
    const kept = trimFrames(src_frames);
    const cover = Cover.init(src_h, src_w, enc.h, enc.w);
    log.info("handoff compile start {d}x{d}x{d} -> {d}x{d}x{d}", .{
        src_w,
        src_h,
        src_t,
        enc.w,
        enc.h,
        kept,
    });

    const idx = try allocator.alloc(i32, kept);
    defer allocator.free(idx);
    var f: u32 = 0;
    while (f < kept) : (f += 1) idx[f] = @intCast(taeh3.takeFrameIndex(f, src_t));

    const wh = try allocator.alloc(f32, cover.resized_w * src_w);
    defer allocator.free(wh);
    const wv = try allocator.alloc(f32, cover.resized_h * src_h);
    defer allocator.free(wv);
    fillAa1d(wh, src_w, cover.resized_w);
    fillAa1d(wv, src_h, cover.resized_h);

    const model: Model = .{
        .idx = .init(.{ .t = kept }, .i32),
        .wh = .init(.{ .wo = cover.resized_w, .w = src_w }, .f32),
        .wv = .init(.{ .ho = cover.resized_h, .h = src_h }, .f32),
        .crop_top = cover.top,
        .crop_left = cover.left,
        .out_h = cover.out_h,
        .out_w = cover.out_w,
    };
    log.info("handoff compile graph tokens={d}", .{kept});
    const exe = try zml.FnExe(handoff).compile(allocator, io, platform, .{
        .shardings = shardings,
        .program_name = "minimax_h3_draft_handoff",
    }, .{.{
        .model = model,
        .rgb = .init(.{ .n = src_t, .c = 3, .h = src_h, .w = src_w }, .f32),
    }});
    errdefer exe.deinit();
    log.info("handoff compiled graph", .{});

    const bufs: zml.Bufferized(Model) = .{
        .idx = try weights.fromItems(io, platform, .init(.{ .t = kept }, .i32), idx),
        .wh = try weights.fromItems(io, platform, .init(.{ .wo = cover.resized_w, .w = src_w }, .f32), wh),
        .wv = try weights.fromItems(io, platform, .init(.{ .ho = cover.resized_h, .h = src_h }, .f32), wv),
    };
    log.info("compile handoff {d}x{d}x{d} -> {d}x{d}x{d} aa={d}x{d}", .{
        src_w,
        src_h,
        src_t,
        cover.out_w,
        cover.out_h,
        kept,
        cover.resized_w,
        cover.resized_h,
    });
    out.* = .{
        .forward = exe,
        .bufs = bufs,
        .frames = kept,
        .height = cover.out_h,
        .width = cover.out_w,
    };
    out.matches_cpu = try checkAgainstCpu(allocator, io, platform, out, src_t, src_h, src_w, src_frames, target_w, target_h);
    if (!out.matches_cpu) return error.HandoffMismatch;
    return out;
}

fn fillPattern(dst: []f32) void {
    var i: usize = 0;
    while (i < dst.len) : (i += 1) {
        const x = @as(f64, @floatFromInt(i % 997)) * 0.01;
        dst[i] = @floatCast(@sin(x) * 0.45 + 0.5);
    }
}

fn cosineMax(a: []const f32, b: []const f32) struct { cos: f64, max: f64 } {
    const n = @min(a.len, b.len);
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    var mx: f64 = 0;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const x: f64 = a[i];
        const y: f64 = b[i];
        dot += x * y;
        na += x * x;
        nb += y * y;
        mx = @max(mx, @abs(x - y));
    }
    const den = @sqrt(na) * @sqrt(nb);
    return .{ .cos = if (den == 0) 0 else dot / den, .max = mx };
}

fn checkAgainstCpu(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    exe: *const Compiled,
    src_t: u32,
    src_h: u32,
    src_w: u32,
    src_frames: u32,
    target_w: u32,
    target_h: u32,
) !bool {
    const raw_n = @as(usize, src_t) * 3 * src_h * src_w;
    const raw = try allocator.alloc(f32, raw_n);
    defer allocator.free(raw);
    fillPattern(raw);

    var rgb = try weights.fromItems(io, platform, .init(.{ .n = src_t, .c = 3, .h = src_h, .w = src_w }, .f32), raw);
    defer rgb.deinit();
    var pixels = try exe.run(allocator, io, rgb);
    defer pixels.deinit();
    const sl = try pixels.toSliceAlloc(allocator, io);
    defer sl.free(allocator);
    const gpu: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, sl.data()));

    const cthw = try taeh3.takeFrames(allocator, raw, src_t, src_h, src_w, src_frames);
    defer allocator.free(cthw);
    const cpu = try prepareDraft(allocator, cthw, src_frames, src_h, src_w, target_w, target_h);
    defer cpu.deinit(allocator);

    if (gpu.len != cpu.rgb.len) {
        log.err("handoff gate size gpu={d} cpu={d}", .{ gpu.len, cpu.rgb.len });
        return false;
    }
    const got = cosineMax(gpu, cpu.rgb);
    const ok = got.cos >= 0.999;
    log.info("handoff gate cos={d:.6} max={d:.5} ok={}", .{ got.cos, got.max, ok });
    return ok;
}
