const std = @import("std");

const zml = @import("zml");

const load = @import("refine/load.zig");
const sku = @import("recipe/sku.zig");
const taehv = @import("refine/taehv.zig");
const weights = @import("recipe/weights.zig");

const log = std.log.scoped(.taehv_check);

// =============================================================================
// taehv_check.zig — GPU oracle: chunked TAEHV vs full
//
// Compare t=31 chunked (16/8) against a single decode. Idle GPU.
// =============================================================================

pub const std_options: std.Options = .{
    .log_level = .info,
};

const latent_t: u32 = 31;
const chunk_t: u32 = 16;
const latent_h: u32 = 4;
const latent_w: u32 = 4;
const channels: u32 = 128;

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    if (init.environ_map.get("BUILD_WORKING_DIRECTORY")) |cwd| {
        var dir = try std.Io.Dir.openDirAbsolute(init.io, cwd, .{});
        defer dir.close(init.io);
        try std.process.setCurrentDir(init.io, dir);
    }
    const io = init.io;
    sku.applyCompileCache(io);
    sku.applyXlaAccelFlags();

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);
    log.info("\n{f}", .{platform.fmtVerbose()});
    const shardings = [_]zml.Sharding{platform.replicated_sharding};

    const path = load.firstExisting(io, &.{
        taehv.default_path,
        "/var/models/super-accel/ltx/taehv/taeltx2_3_wide.safetensors",
    }) orelse return error.TaehvMissing;
    var store = try load.Store.open(allocator, io, path);
    defer store.deinit();

    var progress = std.Progress.start(io, .{ .root_name = "taehv_check" });
    defer progress.end();
    const chunk_m = taehv.Model.init(store.view(), 1, chunk_t);
    var chunk_exe = try taehv.compile(allocator, io, platform, chunk_m, latent_h, latent_w, &shardings, &store.store, &progress, null);
    defer chunk_exe.deinit();
    const full_m = taehv.Model.init(store.view(), 1, latent_t);
    var full_exe = try taehv.compile(allocator, io, platform, full_m, latent_h, latent_w, &shardings, &store.store, &progress, &chunk_exe);
    defer full_exe.deinit();

    const host = try allocator.alloc(f32, channels * latent_t * latent_h * latent_w);
    defer allocator.free(host);
    fillRamp(host);

    var full_lat = try weights.fromItems(io, platform, .init(.{ .n = 1, .c = 128, .t = latent_t, .h = latent_h, .w = latent_w }, .f32), host);
    var chunk_lat = try weights.fromItems(io, platform, .init(.{ .n = 1, .c = 128, .t = latent_t, .h = latent_h, .w = latent_w }, .f32), host);

    var full_run = try zml.FnExe(taehv.decode).Runner(.{.model}).init(&full_exe.decode, allocator, .{ .model = full_exe.bufs });
    defer full_run.deinit(allocator);
    var chunk_run = try zml.FnExe(taehv.decode).Runner(.{.model}).init(&chunk_exe.decode, allocator, .{ .model = chunk_exe.bufs });
    defer chunk_run.deinit(allocator);

    const full = try taehv.decodeLatentWith(allocator, io, platform, &full_exe, &full_run, &full_lat, latent_t, latent_h, latent_w);
    defer allocator.free(full);
    const chunked = try taehv.decodeLatentWith(allocator, io, platform, &chunk_exe, &chunk_run, &chunk_lat, latent_t, latent_h, latent_w);
    defer allocator.free(chunked);

    const keep = taehv.outFrames(latent_t);
    const plane = (latent_h * 32) * (latent_w * 32);
    const first = taehv.outFrames(chunk_t);
    const mae_head = meanAbsRange(full, chunked, keep, plane, 0, first);
    const mae_tail = meanAbsRange(full, chunked, keep, plane, first, keep);
    const mae_all = meanAbsRange(full, chunked, keep, plane, 0, keep);
    const interior = frameDelta(chunked, keep, plane, 60);
    const seam = frameDelta(chunked, keep, plane, first);
    const full_mean = meanAbsVal(full);
    const chunk_mean = meanAbsVal(chunked);

    log.info(
        "frames={d} first_window={d} mae_head={d:.6} mae_tail={d:.6} mae_all={d:.6} d_interior={d:.6} d_seam={d:.6} mean_full={d:.4} mean_chunk={d:.4}",
        .{ keep, first, mae_head, mae_tail, mae_all, interior, seam, full_mean, chunk_mean },
    );

    if (std.math.isNan(mae_all) or std.math.isNan(chunk_mean)) return error.TaehvNan;
    if (chunk_mean > 0.99) return error.TaehvWhite;
    if (chunk_mean < 0.001) return error.TaehvBlack;
    if (mae_head > 1e-3) return error.TaehvHeadMismatch;
    if (seam > interior * 20 + 0.05) return error.TaehvSeam;
    log.info("taehv stitch check ok", .{});
}

fn fillRamp(host: []f32) void {
    const hw = latent_h * latent_w;
    var c: u32 = 0;
    while (c < channels) : (c += 1) {
        var t: u32 = 0;
        while (t < latent_t) : (t += 1) {
            const v = @as(f32, @floatFromInt(t + 1)) * 0.08;
            const off = (c * latent_t + t) * hw;
            @memset(host[off..][0..hw], v);
        }
    }
}

fn meanAbsRange(a: []const f32, b: []const f32, keep: u32, plane: u32, start: u32, end: u32) f64 {
    var sum: f64 = 0;
    var n: u64 = 0;
    var f = start;
    while (f < end) : (f += 1) {
        var ch: u32 = 0;
        while (ch < 3) : (ch += 1) {
            const ia = (ch * keep + f) * plane;
            for (a[ia..][0..plane], b[ia..][0..plane]) |x, y| {
                sum += @abs(x - y);
                n += 1;
            }
        }
    }
    return if (n == 0) 0 else sum / @as(f64, @floatFromInt(n));
}

fn frameDelta(rgb: []const f32, keep: u32, plane: u32, at: u32) f64 {
    if (at == 0 or at >= keep) return 0;
    var sum: f64 = 0;
    var n: u64 = 0;
    var ch: u32 = 0;
    while (ch < 3) : (ch += 1) {
        const ia = (ch * keep + (at - 1)) * plane;
        const ib = (ch * keep + at) * plane;
        for (rgb[ia..][0..plane], rgb[ib..][0..plane]) |x, y| {
            sum += @abs(x - y);
            n += 1;
        }
    }
    return if (n == 0) 0 else sum / @as(f64, @floatFromInt(n));
}

fn meanAbsVal(rgb: []const f32) f64 {
    var sum: f64 = 0;
    for (rgb) |x| sum += @abs(x);
    return if (rgb.len == 0) 0 else sum / @as(f64, @floatFromInt(rgb.len));
}
