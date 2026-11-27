//! Exact-length form of LakonLab v0.3's Sol window/global attention rows.
//! Adapted from sol_h3/model.py, Copyright (c) 2026 Hansheng Chen (Apache-2.0),
//! originally based on Efficient-Large-Model/Sol-H3-VAE@6a8a33f.

const std = @import("std");

pub const period = 5;
pub const window = 16;
pub const globals = 5;

pub const Grid = struct {
    t: usize,
    h: usize,
    w: usize,

    pub fn groups(self: Grid) usize {
        return (self.t + period - 1) / period;
    }

    pub fn videoRows(self: Grid) usize {
        return self.t * self.h * self.w;
    }

    pub fn rows(self: Grid) usize {
        return self.videoRows() + self.groups() * globals;
    }

    pub fn frames(self: Grid) usize {
        return 4 * self.t - 3 * self.groups();
    }
};

pub const Bucket = struct {
    nq: usize,
    nk: usize,
    // Both arrays use the same sequence order, with no padding or sentinels.
    q: std.ArrayList(i32) = .empty,
    k: std.ArrayList(i32) = .empty,
};

pub const Plan = struct {
    buckets: std.ArrayList(Bucket) = .empty,

    pub fn deinit(self: *Plan, allocator: std.mem.Allocator) void {
        for (self.buckets.items) |*b| {
            b.q.deinit(allocator);
            b.k.deinit(allocator);
        }
        self.buckets.deinit(allocator);
    }

    fn sequence(self: *Plan, a: std.mem.Allocator, nq: usize, nk: usize) !struct { []i32, []i32 } {
        const bucket = for (self.buckets.items) |*b| {
            if (b.nq == nq and b.nk == nk) break b;
        } else blk: {
            const b = try self.buckets.addOne(a);
            b.* = .{ .nq = nq, .nk = nk };
            break :blk b;
        };
        return .{ try bucket.q.addManyAsSlice(a, nq), try bucket.k.addManyAsSlice(a, nk) };
    }

    /// Global queries see their native temporal group's entire real grid and its five globals.
    pub fn global(a: std.mem.Allocator, grid: Grid) !Plan {
        var p: Plan = .{};
        errdefer p.deinit(a);
        for (0..grid.groups()) |g| {
            const lo = g * period * grid.h * grid.w;
            const hi = @min((g + 1) * period, grid.t) * grid.h * grid.w;
            const q, const k = try p.sequence(a, globals, hi - lo + globals);
            fillRange(q, grid.videoRows() + g * globals);
            fillRange(k[0 .. hi - lo], lo);
            @memcpy(k[hi - lo ..], q);
        }
        return p;
    }

    /// Intersect each padded window with the real video before building any rows.
    /// A shifted temporal window straddles at most two *different* native groups.
    pub fn windows(a: std.mem.Allocator, grid: Grid, shifted: bool) !Plan {
        var p: Plan = .{};
        errdefer p.deinit(a);
        const st: usize = if (shifted) 2 else 0;
        const s: usize = if (shifted) window / 2 else 0;
        var t0: usize = 0;
        while (t0 < grid.t) {
            const t1 = @min(grid.t, t0 + period - (t0 + st) % period);
            const g0 = t0 / period;
            const g1 = (t1 - 1) / period;
            var y0: usize = 0;
            while (y0 < grid.h) {
                const y1 = @min(grid.h, y0 + window - (y0 + s) % window);
                var x0: usize = 0;
                while (x0 < grid.w) {
                    const x1 = @min(grid.w, x0 + window - (x0 + s) % window);
                    const nq = (t1 - t0) * (y1 - y0) * (x1 - x0);
                    const q, const k = try p.sequence(a, nq, nq + (g1 - g0 + 1) * globals);
                    var i: usize = 0;
                    for (t0..t1) |t| {
                        for (y0..y1) |y| {
                            fillRange(q[i..][0 .. x1 - x0], (t * grid.h + y) * grid.w + x0);
                            i += x1 - x0;
                        }
                    }
                    @memcpy(k[0..nq], q);
                    fillRange(k[nq..], grid.videoRows() + g0 * globals);
                    x0 = x1;
                }
                y0 = y1;
            }
            t0 = t1;
        }
        return p;
    }
};

fn fillRange(out: []i32, start: usize) void {
    for (out, start..) |*x, n| x.* = @intCast(n);
}
