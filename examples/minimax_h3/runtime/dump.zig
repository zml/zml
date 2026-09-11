const std = @import("std");
const zml = @import("zml");

const log = std.log.scoped(.minimax_h3_dump);

var path_buf: [4096]u8 = undefined;
var path_len: usize = 0;

pub fn resolve(cli: []const u8) []const u8 {
    if (cli.len != 0) return cli;
    if (envPath("H3_DUMP")) |p| return p;
    if (envPath("H3_LAYER_DUMP")) |p| return p;
    return "";
}

fn envPath(name: [:0]const u8) ?[]const u8 {
    const raw = std.c.getenv(name) orelse return null;
    const path = std.mem.span(raw);
    return if (path.len == 0) null else path;
}

pub fn setPath(path: []const u8) void {
    const n = @min(path.len, path_buf.len);
    @memcpy(path_buf[0..n], path[0..n]);
    path_len = n;
}

pub fn enabled() bool {
    return path_len != 0;
}

pub fn location() []const u8 {
    return path_buf[0..path_len];
}

pub fn open(io: std.Io) !std.Io.Dir {
    const p = location();
    try std.Io.Dir.cwd().createDirPath(io, p);
    if (std.fs.path.isAbsolute(p)) return std.Io.Dir.openDirAbsolute(io, p, .{});
    return std.Io.Dir.cwd().openDir(io, p, .{});
}

pub fn writeBytes(io: std.Io, dir: std.Io.Dir, name: []const u8, bytes: []const u8) !void {
    const file = try dir.createFile(io, name, .{});
    defer file.close(io);
    var writer = file.writer(io, &.{});
    try writer.interface.writeAll(bytes);
}

pub fn writeShape(io: std.Io, dir: std.Io.Dir, name: []const u8, dims: []const i64) !void {
    var buf: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    for (dims, 0..) |d, i| {
        if (i != 0) try w.writeByte(' ');
        try w.print("{d}", .{d});
    }
    var name_buf: [128]u8 = undefined;
    const shape_name = try std.fmt.bufPrint(&name_buf, "{s}.shape", .{name});
    try writeBytes(io, dir, shape_name, w.buffered());
}

pub const Stats = struct {
    n: usize,
    finite: usize,
    nan: usize,
    inf: usize,
    min: f32,
    max: f32,
    mean: f64,
    std: f64,
};

pub fn statsF32(values: []const f32) Stats {
    var nan_n: usize = 0;
    var inf_n: usize = 0;
    var finite_n: usize = 0;
    var min: f32 = std.math.floatMax(f32);
    var max: f32 = -std.math.floatMax(f32);
    var sum: f64 = 0;
    var sumsq: f64 = 0;
    for (values) |v| {
        if (std.math.isNan(v)) {
            nan_n += 1;
            continue;
        }
        if (std.math.isInf(v)) {
            inf_n += 1;
            continue;
        }
        finite_n += 1;
        min = @min(min, v);
        max = @max(max, v);
        const d: f64 = v;
        sum += d;
        sumsq += d * d;
    }
    const mean = if (finite_n == 0) 0 else sum / @as(f64, @floatFromInt(finite_n));
    const var_ = if (finite_n == 0) 0 else sumsq / @as(f64, @floatFromInt(finite_n)) - mean * mean;
    return .{
        .n = values.len,
        .finite = finite_n,
        .nan = nan_n,
        .inf = inf_n,
        .min = if (finite_n == 0) 0 else min,
        .max = if (finite_n == 0) 0 else max,
        .mean = mean,
        .std = if (var_ > 0) @sqrt(var_) else 0,
    };
}

pub fn logF32(name: []const u8, values: []const f32, dims: []const i64) void {
    const s = statsF32(values);
    log.info(
        "tensor {s} shape={any} n={d} finite={d} nan={d} inf={d} min={d:.6} max={d:.6} mean={d:.6} std={d:.6}",
        .{ name, dims, s.n, s.finite, s.nan, s.inf, s.min, s.max, s.mean, s.std },
    );
    if (s.nan != 0 or s.inf != 0)
        log.err("tensor {s} NON-FINITE nan={d} inf={d}", .{ name, s.nan, s.inf });
}

pub fn f32s(io: std.Io, name: []const u8, values: []const f32, dims: []const i64) !void {
    if (!enabled()) return;
    var dir = try open(io);
    defer dir.close(io);
    var name_buf: [128]u8 = undefined;
    const file_name = try std.fmt.bufPrint(&name_buf, "{s}.f32", .{name});
    try writeBytes(io, dir, file_name, std.mem.sliceAsBytes(values));
    try writeShape(io, dir, name, dims);
    logF32(name, values, dims);
}

pub fn u32s(io: std.Io, name: []const u8, values: []const u32, dims: []const i64) !void {
    if (!enabled()) return;
    var dir = try open(io);
    defer dir.close(io);
    var name_buf: [128]u8 = undefined;
    const file_name = try std.fmt.bufPrint(&name_buf, "{s}.u32", .{name});
    try writeBytes(io, dir, file_name, std.mem.sliceAsBytes(values));
    try writeShape(io, dir, name, dims);
    if (values.len == 0) {
        log.info("tensor {s} shape={any} n=0", .{ name, dims });
        return;
    }
    var min = values[0];
    var max = values[0];
    var sum: f64 = 0;
    for (values) |v| {
        min = @min(min, v);
        max = @max(max, v);
        sum += @floatFromInt(v);
    }
    log.info(
        "tensor {s} shape={any} n={d} min={d} max={d} mean={d:.4}",
        .{ name, dims, values.len, min, max, sum / @as(f64, @floatFromInt(values.len)) },
    );
}

pub fn rgbU8AsF32(allocator: std.mem.Allocator, io: std.Io, name: []const u8, rgb: []const u8, h: u32, w: u32) !void {
    if (!enabled()) return;
    const out = try allocator.alloc(f32, rgb.len);
    defer allocator.free(out);
    for (rgb, out) |b, *d| d.* = @as(f32, @floatFromInt(b)) / 255.0;
    try f32s(io, name, out, &.{ @intCast(h), @intCast(w), 3 });
}

pub fn text(io: std.Io, name: []const u8, body: []const u8) !void {
    if (!enabled()) return;
    var dir = try open(io);
    defer dir.close(io);
    try writeBytes(io, dir, name, body);
}

fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

pub fn buffer(allocator: std.mem.Allocator, io: std.Io, name: []const u8, buf: *zml.Buffer) !void {
    if (!enabled()) return;
    const slice = try buf.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    const shape = buf.shape();
    const n = shape.count();
    const values = try allocator.alloc(f32, n);
    defer allocator.free(values);
    const raw = slice.data();
    switch (shape.dtype()) {
        .f32 => {
            const src = std.mem.bytesAsSlice(f32, raw[0 .. n * 4]);
            @memcpy(values, src);
        },
        .bf16 => {
            const src = std.mem.bytesAsSlice(u16, raw[0 .. n * 2]);
            for (src, values) |b, *d| d.* = bf16ToF32(b);
        },
        .f16 => {
            const src = std.mem.bytesAsSlice(u16, raw[0 .. n * 2]);
            for (src, values) |b, *d| d.* = @floatCast(@as(f16, @bitCast(b)));
        },
        else => {
            log.warn("dump {s}: skip dtype {s}", .{ name, @tagName(shape.dtype()) });
            return;
        },
    }
    try f32s(io, name, values, shape.dims());
}
