//! Tests of `zml.moe.megamoe.weights` (checkpoint weights to the MegaMoE kernel layouts).
//!
//! The index-map tests run on the host. The device tests are opt-in (they need a PJRT device;
//! this host has no CPU plugin) and compare bytes:
//!   MEGAMOE_WEIGHTS_RANDOM=1: every transform on small random tensors against host references
//!     scattered with the index maps (vLLM's loader formulation), incl. block-scale expansion.
//!   MEGAMOE_WEIGHTS_STORES=<dir>[,<dir>...]: reference weight stores (`S/dumps/weights/<key>`,
//!     layouts in `LAYOUTS.md` §4). `real_*`: the checkpoint experts `ckpt.experts.<g>.*` of
//!     each rank file (checkpoint dtypes) against those experts' slices of the kernel-layout
//!     tensors written by vLLM's loader, and `ckpt.shared_experts.*` (32x32 block scales) against
//!     the shared tensors. `rand_*`: the `*_raw` tensors (all experts, 1x32 scales) against
//!     DeepGEMM's `transform_weights_for_mega_moe` outputs.
//!   MEGAMOE_WEIGHTS_SHARDED=<real_* store dir> (two devices): both ranks' experts sharded by
//!     expert; no collective in the optimized HLO, outputs byte for byte.
//!   MEGAMOE_WEIGHTS_E2E=<dump dir>[,...]: single-rank DeepGEMM dumps whose weight store has the
//!     raw tensors: transforms and `forwardStaged` in one program, `y` bit for bit.
//!
//!   bazel test --config=remote --config=release --@zml//platforms:cuda=true \
//!     --repo_env=RULES_ZIG_CACHE_PREFIX_LINUX=/tmp/benjamin-zig-cache \
//!     --test_strategy=standalone --test_env=CUDA_VISIBLE_DEVICES \
//!     --test_env=MEGAMOE_WEIGHTS_RANDOM=1 --test_env=MEGAMOE_WEIGHTS_STORES=<dir>,<dir> \
//!     //zml/kernels/moe/megamoe:weights_test
const std = @import("std");

const zml = @import("zml");
const megamoe = zml.moe.megamoe;
const weights = megamoe.weights;

const Tensor = zml.Tensor;

// ==================== index maps (host only) ====================

test "l1ScaleRow is the gate/up interleave followed by the UTCCP transpose" {
    for (0..2304) |r| {
        for (0..2) |h| {
            const row: u32 = @intCast(r);
            const half: u1 = @intCast(h);
            try std.testing.expectEqual(weights.utccpRow(weights.l1WeightRow(row, half)), weights.l1ScaleRow(row, half));
        }
    }
}

test "index maps follow DeepGEMM's and vLLM's reshapes" {
    const n = 256; // rows of w1 and of w3
    // DeepGEMM `_interleave_weights`: stack([gate.reshape(n/8, 8), up.reshape(n/8, 8)], dim=1).
    for (0..n / 8) |b| for (0..2) |h| for (0..8) |m| {
        const src: u32 = @intCast(8 * b + m);
        try std.testing.expectEqual(@as(u32, @intCast((2 * b + h) * 8 + m)), weights.l1WeightRow(src, @intCast(h)));
    };
    // DeepGEMM `_transpose_sf_for_utccp`: reshape(-1, 4, 32).transpose(1, 2), and vLLM's L2
    // `dst.view(-1, 32, 4).transpose(1, 2) <- src.reshape(-1, 4, 32)`.
    for (0..2 * n / 128) |b| for (0..4) |j| for (0..32) |i| {
        const src: u32 = @intCast(128 * b + 32 * j + i);
        try std.testing.expectEqual(@as(u32, @intCast(128 * b + 4 * i + j)), weights.utccpRow(src));
    };
    // vLLM's L1 scales: `dst.view(-1, 2, 2, 8, 4)[:, :, half].permute(0, 3, 1, 2, 4)
    // <- src.reshape(-1, 4, 2, 8)`, i.e. dst (b, q, half, m, j) <- src (b, j, q, m).
    for (0..n / 64) |b| for (0..4) |j| for (0..2) |q| for (0..8) |m| for (0..2) |h| {
        const src: u32 = @intCast(64 * b + 16 * j + 8 * q + m);
        const dst: u32 = @intCast((((b * 2 + q) * 2 + h) * 8 + m) * 4 + j);
        try std.testing.expectEqual(dst, weights.l1ScaleRow(src, @intCast(h)));
    };
}

test "index maps are permutations" {
    const n = 2304;
    var seen_w: [2 * n]bool = @splat(false);
    var seen_sf: [2 * n]bool = @splat(false);
    for (0..n) |r| for (0..2) |h| {
        const w = weights.l1WeightRow(@intCast(r), @intCast(h));
        const sf = weights.l1ScaleRow(@intCast(r), @intCast(h));
        try std.testing.expect(w < 2 * n and !seen_w[w]);
        try std.testing.expect(sf < 2 * n and !seen_sf[sf]);
        seen_w[w] = true;
        seen_sf[sf] = true;
    };
    var seen_l2: [5120]bool = @splat(false);
    for (0..seen_l2.len) |r| {
        const row = weights.utccpRow(@intCast(r));
        try std.testing.expect(row < seen_l2.len and !seen_l2[row]);
        seen_l2[row] = true;
    }
}

// ==================== device helpers ====================

fn routedL1(in: [4]Tensor) [2]Tensor {
    const r = weights.routedL1(in[0], in[1], in[2], in[3]);
    return .{ r.w, r.sf };
}

fn routedL2(in: [2]Tensor) [2]Tensor {
    const r = weights.routedL2(in[0], in[1]);
    return .{ r.w, r.sf };
}

fn sharedL1(in: [4]Tensor) [2]Tensor {
    const r = weights.sharedL1(in[0], in[1], in[2], in[3]);
    return .{ r.w, r.sf };
}

fn sharedL2(in: [2]Tensor) [2]Tensor {
    const r = weights.sharedL2(in[0], in[1]);
    return .{ r.w, r.sf };
}

/// The random stores keep gate rows then up rows in one tensor (`[E, 2I, ..]`).
fn routedL1Joined(in: [2]Tensor) [2]Tensor {
    const i = @divExact(in[0].dim(1), 2);
    return routedL1(.{ in[0].slice(1, .{ .end = i }), in[1].slice(1, .{ .end = i }), in[0].slice(1, .{ .start = i }), in[1].slice(1, .{ .start = i }) });
}

/// Same for the shared expert (`[2I, ..]`).
fn sharedL1Joined(in: [2]Tensor) [2]Tensor {
    const i = @divExact(in[0].dim(0), 2);
    return sharedL1(.{ in[0].slice(0, .{ .end = i }), in[1].slice(0, .{ .end = i }), in[0].slice(0, .{ .start = i }), in[1].slice(0, .{ .start = i }) });
}

/// A host tensor: its shape (with the dtype the transform receives) and its bytes.
const Input = struct { shape: zml.Shape, bytes: []const u8 };

/// Runs `func` on the device(s) and returns its two outputs' bytes (whole tensors, gathered from
/// their shards); the caller frees them.
fn runTransform(comptime n: usize, comptime func: fn ([n]Tensor) [2]Tensor, inputs: [n]Input, opts: zml.Compiler.Options) ![2][]u8 {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var tensors: [n]Tensor = undefined;
    var buffers: [n]zml.Buffer = undefined;
    var loaded: usize = 0;
    defer for (buffers[0..loaded]) |*b| b.deinit();
    for (inputs, &tensors, &buffers) |input, *t, *b| {
        std.debug.assert(input.shape.byteSize() == input.bytes.len);
        t.* = .fromShape(input.shape);
        b.* = try .fromBytes(io, platform, input.shape, input.bytes);
        loaded += 1;
    }
    var exe = try platform.compileFn(allocator, io, func, .{tensors}, opts);
    defer exe.deinit();
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);
    var outputs: [2]zml.Buffer = undefined;
    runner.run(io, .{buffers}, &outputs, .{ .wait = true });
    defer for (&outputs) |*o| o.deinit();
    var result: [2][]u8 = undefined;
    for (&result, outputs, 0..) |*r, o, i| {
        const host = try o.toSliceAlloc(allocator, io);
        defer host.free(allocator);
        r.* = allocator.dupe(u8, host.constData()) catch |err| {
            if (i == 1) allocator.free(result[0]);
            return err;
        };
    }
    return result;
}

/// Prints the byte mismatch count of `actual` against `expected` and returns it (a size
/// difference counts every byte).
fn countMismatches(label: []const u8, name: []const u8, actual: []const u8, expected: []const u8) usize {
    var mismatched: usize = 0;
    var first: ?usize = null;
    if (actual.len != expected.len) {
        mismatched = @max(actual.len, expected.len);
    } else for (actual, expected, 0..) |a, e, i| {
        if (a == e) continue;
        mismatched += 1;
        if (first == null) first = i;
    }
    std.debug.print("{s} {s}: {d}/{d} bytes differ", .{ label, name, mismatched, expected.len });
    if (actual.len != expected.len) std.debug.print(" (size {d}, expected {d})", .{ actual.len, expected.len });
    if (first) |i| std.debug.print(" (first at byte {d}: 0x{x:0>2}, expected 0x{x:0>2})", .{ i, actual[i], expected[i] });
    std.debug.print("\n", .{});
    return mismatched;
}

/// Runs a transform and compares both outputs; returns the total byte mismatch count.
fn check(comptime n: usize, comptime func: fn ([n]Tensor) [2]Tensor, label: []const u8, names: [2][]const u8, inputs: [n]Input, expected: [2][]const u8) !usize {
    return checkWith(n, func, label, names, inputs, expected, .{});
}

fn checkWith(comptime n: usize, comptime func: fn ([n]Tensor) [2]Tensor, label: []const u8, names: [2][]const u8, inputs: [n]Input, expected: [2][]const u8, opts: zml.Compiler.Options) !usize {
    const allocator = std.testing.allocator;
    const result = try runTransform(n, func, inputs, opts);
    defer for (result) |r| allocator.free(r);
    return countMismatches(label, names[0], result[0], expected[0]) + countMismatches(label, names[1], result[1], expected[1]);
}

// ==================== small random tensors vs host references ====================

/// Host reference of the routed L1 transform for `e` experts of `n` rows (vLLM's scatter):
/// row `r` of w1/w3 goes to `l1WeightRow`, its scale bytes to word `kq` of `l1ScaleRow`.
fn refL1(allocator: std.mem.Allocator, e: usize, n: usize, w: [2][]const u8, sf: [2][]const u8) ![2][]u8 {
    const row_bytes = w[0].len / (e * n);
    const groups = sf[0].len / (e * n);
    const out_w = try allocator.alloc(u8, 2 * w[0].len);
    errdefer allocator.free(out_w);
    const out_sf = try allocator.alloc(u8, 2 * sf[0].len);
    for (0..e) |x| for (0..n) |r| for (0..2) |h| {
        const dst_w = weights.l1WeightRow(@intCast(r), @intCast(h));
        @memcpy(out_w[(x * 2 * n + dst_w) * row_bytes ..][0..row_bytes], w[h][(x * n + r) * row_bytes ..][0..row_bytes]);
        const dst_sf = weights.l1ScaleRow(@intCast(r), @intCast(h));
        for (0..groups) |g| out_sf[((x * (groups / 4) + g / 4) * 2 * n + dst_sf) * 4 + g % 4] = sf[h][(x * n + r) * groups + g];
    };
    return .{ out_w, out_sf };
}

/// Host reference of the L2 scale transform: row `r` to `utccpRow(r)`, MN-major words.
fn refL2Scales(allocator: std.mem.Allocator, e: usize, n: usize, sf: []const u8) ![]u8 {
    const groups = sf.len / (e * n);
    const out = try allocator.alloc(u8, sf.len);
    for (0..e) |x| for (0..n) |r| for (0..groups) |g| {
        out[((x * (groups / 4) + g / 4) * n + weights.utccpRow(@intCast(r))) * 4 + g % 4] = sf[(x * n + r) * groups + g];
    };
    return out;
}

/// Block scales `[sr, sc]` repeated to `[rows, cols]` (vLLM `_pack_scale`'s repeat_interleave).
fn refExpand(allocator: std.mem.Allocator, scale: []const u8, sr: usize, sc: usize, rows: usize, cols: usize) ![]u8 {
    const out = try allocator.alloc(u8, rows * cols);
    for (0..rows) |r| for (0..cols) |c| {
        out[r * cols + c] = scale[(r / (rows / sr)) * sc + c / (cols / sc)];
    };
    return out;
}

fn randomBytes(allocator: std.mem.Allocator, random: std.Random, len: usize) ![]u8 {
    const out = try allocator.alloc(u8, len);
    random.bytes(out);
    return out;
}

test "transforms on random tensors match the index-map references" {
    if (std.c.getenv("MEGAMOE_WEIGHTS_RANDOM") == null) return error.SkipZigTest;
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(0x3e6a);
    const random = prng.random();

    // E experts, I intermediate, H hidden: 2I = 4 and H = 3 UTCCP blocks, K/128 = 3 and 2 words.
    const E = 3;
    const I = 256;
    const H = 384;
    var mismatched: usize = 0;

    // Routed, with the checkpoint dtypes (i8 packed e2m1, f8e8m0 scales).
    {
        const w = [2][]const u8{ try randomBytes(arena, random, E * I * H / 2), try randomBytes(arena, random, E * I * H / 2) };
        const sf = [2][]const u8{ try randomBytes(arena, random, E * I * H / 32), try randomBytes(arena, random, E * I * H / 32) };
        const ref = try refL1(arena, E, I, w, sf);
        const w_shape: zml.Shape = .init(.{ E, I, H / 2 }, .i8);
        const sf_shape: zml.Shape = .init(.{ E, I, H / 32 }, .f8e8m0);
        mismatched += try check(4, routedL1, "random routedL1", .{ "l1_w", "l1_sf" }, .{
            .{ .shape = w_shape, .bytes = w[0] },
            .{ .shape = sf_shape, .bytes = sf[0] },
            .{ .shape = w_shape, .bytes = w[1] },
            .{ .shape = sf_shape, .bytes = sf[1] },
        }, ref);
    }
    {
        const w = try randomBytes(arena, random, E * H * I / 2);
        const sf = try randomBytes(arena, random, E * H * I / 32);
        mismatched += try check(2, routedL2, "random routedL2", .{ "l2_w", "l2_sf" }, .{
            .{ .shape = .init(.{ E, H, I / 2 }, .u8), .bytes = w },
            .{ .shape = .init(.{ E, H, I / 32 }, .u8), .bytes = sf },
        }, .{ w, try refL2Scales(arena, E, H, sf) });
    }

    // Shared: 32x32 blocks (the checkpoint), 1x32 (no expansion) and a 4x64 tiling.
    for ([_][2]usize{ .{ 32, 32 }, .{ 1, 32 }, .{ 4, 64 } }) |block| {
        const sr = I / block[0];
        const sc = H / block[1];
        const w = [2][]const u8{ try randomBytes(arena, random, I * H), try randomBytes(arena, random, I * H) };
        const blocks = [2][]const u8{ try randomBytes(arena, random, sr * sc), try randomBytes(arena, random, sr * sc) };
        const full = [2][]const u8{ try refExpand(arena, blocks[0], sr, sc, I, H / 32), try refExpand(arena, blocks[1], sr, sc, I, H / 32) };
        const ref = try refL1(arena, 1, I, w, full);
        const label = try std.fmt.allocPrint(arena, "random sharedL1 ({d}x{d} scales)", .{ block[0], block[1] });
        const w_shape: zml.Shape = .init(.{ I, H }, .f8e4m3fn);
        const sf_shape: zml.Shape = .init(.{ sr, sc }, .f8e8m0);
        mismatched += try check(4, sharedL1, label, .{ "shared_l1_w", "shared_l1_sf" }, .{
            .{ .shape = w_shape, .bytes = w[0] },
            .{ .shape = sf_shape, .bytes = blocks[0] },
            .{ .shape = w_shape, .bytes = w[1] },
            .{ .shape = sf_shape, .bytes = blocks[1] },
        }, ref);
    }
    {
        const sr = H / 32;
        const sc = I / 32;
        const w = try randomBytes(arena, random, H * I);
        const blocks = try randomBytes(arena, random, sr * sc);
        const full = try refExpand(arena, blocks, sr, sc, H, I / 32);
        mismatched += try check(2, sharedL2, "random sharedL2 (32x32 scales)", .{ "shared_l2_w", "shared_l2_sf" }, .{
            .{ .shape = .init(.{ H, I }, .u8), .bytes = w },
            .{ .shape = .init(.{ sr, sc }, .u8), .bytes = blocks },
        }, .{ w, try refL2Scales(arena, 1, H, full) });
    }
    try std.testing.expectEqual(@as(usize, 0), mismatched);
}

// ==================== reference weight stores ====================

const Store = struct {
    registry: zml.safetensors.TensorRegistry,

    /// Null when the file does not exist.
    fn open(io: std.Io, path: []const u8) !?Store {
        const file = std.Io.Dir.openFile(.cwd(), io, path, .{ .mode = .read_only }) catch |err| switch (err) {
            error.FileNotFound => return null,
            else => return err,
        };
        defer file.close(io);
        var registry: zml.safetensors.TensorRegistry = .init(std.testing.allocator);
        errdefer registry.deinit();
        try zml.safetensors.parseSafetensors(std.testing.allocator, io, &registry, file);
        return .{ .registry = registry };
    }

    fn deinit(self: *Store) void {
        self.registry.deinit();
    }

    fn has(self: *const Store, name: []const u8) bool {
        return self.registry.tensors.get(name) != null;
    }

    fn tensor(self: *const Store, name: []const u8) !zml.safetensors.Tensor {
        return self.registry.tensors.get(name) orelse {
            std.debug.print("tensor {s} missing\n", .{name});
            return error.MissingTensor;
        };
    }

    /// `len` bytes of tensor `name` from byte `start` of its memory image.
    fn readRange(self: *const Store, allocator: std.mem.Allocator, io: std.Io, name: []const u8, start: u64, len: u64) ![]u8 {
        const t = try self.tensor(name);
        std.debug.assert(start + len <= t.byteSize());
        const out = try allocator.alloc(u8, len);
        errdefer allocator.free(out);
        const file = try std.Io.Dir.openFile(.cwd(), io, t.file_uri, .{ .mode = .read_only });
        defer file.close(io);
        if (try file.readPositionalAll(io, out, t.offset + start) != out.len) return error.EndOfStream;
        return out;
    }

    fn read(self: *const Store, allocator: std.mem.Allocator, io: std.Io, name: []const u8) ![]u8 {
        return self.readRange(allocator, io, name, 0, (try self.tensor(name)).byteSize());
    }

    fn shape(self: *const Store, name: []const u8) !zml.Shape {
        return (try self.tensor(name)).shape;
    }
};

/// A `real_*` rank file's checkpoint experts, stacked along a leading expert axis with the
/// checkpoint dtypes (`inputs`: w1, w1 scale, w3, w3 scale, w2, w2 scale), and their slices of
/// the kernel-layout tensors (`expected`: l1_w, l1_sf, l2_w, l2_sf; local expert = global id -
/// the rank's first expert). Allocated in `arena`.
const RealExperts = struct {
    ids: []const u32,
    inputs: [6]Input,
    expected: [4][]const u8,

    fn load(arena: std.mem.Allocator, dir: []const u8, rank: usize, store: *const Store) !RealExperts {
        const io = std.testing.io;
        const json_path = try std.fmt.allocPrint(arena, "{s}/rank{d}.json", .{ dir, rank });
        const json_bytes = try std.Io.Dir.cwd().readFileAlloc(io, json_path, arena, .limited(1 << 24));
        const info = try std.json.parseFromSliceLeaky(struct { experts: [2]u32 }, arena, json_bytes, .{ .ignore_unknown_fields = true });
        const first_expert = info.experts[0];

        var ids: std.ArrayList(u32) = .empty;
        for (store.registry.tensors.keys()) |key| {
            const prefix = "ckpt.experts.";
            const suffix = ".w1.weight";
            if (!std.mem.startsWith(u8, key, prefix) or !std.mem.endsWith(u8, key, suffix)) continue;
            try ids.append(arena, try std.fmt.parseInt(u32, key[prefix.len .. key.len - suffix.len], 10));
        }
        std.mem.sort(u32, ids.items, {}, std.sort.asc(u32));
        if (ids.items.len == 0) return error.NoCheckpointExperts;
        const n = ids.items.len;

        var result: RealExperts = .{ .ids = ids.items, .inputs = undefined, .expected = undefined };
        const parts = [_][]const u8{ "w1.weight", "w1.scale", "w3.weight", "w3.scale", "w2.weight", "w2.scale" };
        for (parts, &result.inputs) |part, *input| {
            const one = try store.shape(try std.fmt.allocPrint(arena, "ckpt.experts.{d}.{s}", .{ ids.items[0], part }));
            const bytes = try arena.alloc(u8, n * one.byteSize());
            for (ids.items, 0..) |id, j| {
                const t = try store.read(arena, io, try std.fmt.allocPrint(arena, "ckpt.experts.{d}.{s}", .{ id, part }));
                if (t.len != one.byteSize()) return error.UnexpectedShape;
                @memcpy(bytes[j * t.len ..][0..t.len], t);
            }
            const dtype: zml.DataType = if (std.mem.endsWith(u8, part, "weight")) .i8 else .f8e8m0;
            input.* = .{ .shape = .init(.{ @as(i64, @intCast(n)), one.dim(0), one.dim(1) }, dtype), .bytes = bytes };
        }
        // The expert axis leads every kernel-layout memory order.
        for ([_][]const u8{ "l1_w", "l1_sf", "l2_w", "l2_sf" }, &result.expected) |name, *expected| {
            const full = try store.shape(name);
            const per_expert = @divExact(full.byteSize(), @as(u64, @intCast(full.dim(0))));
            const bytes = try arena.alloc(u8, n * per_expert);
            for (ids.items, 0..) |id, j| {
                const slice = try store.readRange(arena, io, name, @as(u64, id - first_expert) * per_expert, per_expert);
                @memcpy(bytes[j * per_expert ..][0..per_expert], slice);
            }
            expected.* = bytes;
        }
        return result;
    }
};

/// The `real_*` rank file: the checkpoint experts it holds against their kernel-layout slices.
fn checkRealRank(dir: []const u8, rank: usize, store: *const Store) !usize {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const experts: RealExperts = try .load(arena, dir, rank, store);
    const label = try std.fmt.allocPrint(arena, "{s} rank {d} experts {any}", .{ std.fs.path.basename(dir), rank, experts.ids });
    const mismatched = try check(4, routedL1, label, .{ "l1_w", "l1_sf" }, experts.inputs[0..4].*, experts.expected[0..2].*);
    return mismatched + try check(2, routedL2, label, .{ "l2_w", "l2_sf" }, experts.inputs[4..6].*, experts.expected[2..4].*);
}

/// The `rand_*` rank file: all experts' raw tensors against the transformed ones.
fn checkRandomRank(dir: []const u8, rank: usize, store: *const Store) !usize {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const label = try std.fmt.allocPrint(allocator, "{s} rank {d} (all experts)", .{ std.fs.path.basename(dir), rank });
    defer allocator.free(label);

    var mismatched: usize = 0;
    {
        const w = try store.read(allocator, io, "l1_w_raw");
        defer allocator.free(w);
        const sf = try store.read(allocator, io, "l1_sf_raw_e8m0");
        defer allocator.free(sf);
        const result = try runTransform(2, routedL1Joined, .{
            .{ .shape = try store.shape("l1_w_raw"), .bytes = w },
            .{ .shape = try store.shape("l1_sf_raw_e8m0"), .bytes = sf },
        }, .{});
        defer for (result) |r| allocator.free(r);
        for ([_][]const u8{ "l1_w", "l1_sf" }, result) |name, actual| {
            const expected = try store.read(allocator, io, name);
            defer allocator.free(expected);
            mismatched += countMismatches(label, name, actual, expected);
        }
    }
    {
        // `transform_weights_for_mega_moe` leaves L2 weights unchanged: `l2_w` is the raw input.
        const w = try store.read(allocator, io, "l2_w");
        defer allocator.free(w);
        const sf = try store.read(allocator, io, "l2_sf_raw_e8m0");
        defer allocator.free(sf);
        const expected = try store.read(allocator, io, "l2_sf");
        defer allocator.free(expected);
        mismatched += try check(2, routedL2, label, .{ "l2_w", "l2_sf" }, .{
            .{ .shape = try store.shape("l2_w"), .bytes = w },
            .{ .shape = try store.shape("l2_sf_raw_e8m0"), .bytes = sf },
        }, .{ w, expected });
    }
    return mismatched;
}

/// `shared.safetensors`: the checkpoint shared expert (`real_*`, 32x32 block scales, checkpoint
/// dtypes) or the raw random one (`rand_*`, 1x32 scales, gate and up rows in one tensor).
fn checkShared(dir: []const u8, store: *const Store) !usize {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const label = try std.fmt.allocPrint(arena, "{s} shared", .{std.fs.path.basename(dir)});

    var expected: [4][]u8 = undefined;
    for ([_][]const u8{ "shared_l1_w", "shared_l1_sf", "shared_l2_w", "shared_l2_sf" }, &expected) |name, *bytes| {
        bytes.* = try store.read(arena, io, name);
    }

    var mismatched: usize = 0;
    if (store.has("ckpt.shared_experts.w1.weight")) {
        var in: [6]Input = undefined;
        for ([_][]const u8{ "w1.weight", "w1.scale", "w3.weight", "w3.scale", "w2.weight", "w2.scale" }, &in) |part, *input| {
            const name = try std.fmt.allocPrint(arena, "ckpt.shared_experts.{s}", .{part});
            const dtype: zml.DataType = if (std.mem.endsWith(u8, part, "weight")) .f8e4m3fn else .f8e8m0;
            input.* = .{ .shape = (try store.shape(name)).withDtype(dtype), .bytes = try store.read(arena, io, name) };
        }
        mismatched += try check(4, sharedL1, label, .{ "shared_l1_w", "shared_l1_sf" }, in[0..4].*, .{ expected[0], expected[1] });
        mismatched += try check(2, sharedL2, label, .{ "shared_l2_w", "shared_l2_sf" }, in[4..6].*, .{ expected[2], expected[3] });
    } else {
        mismatched += try check(2, sharedL1Joined, label, .{ "shared_l1_w", "shared_l1_sf" }, .{
            .{ .shape = try store.shape("shared_l1_w_raw"), .bytes = try store.read(arena, io, "shared_l1_w_raw") },
            .{ .shape = try store.shape("shared_l1_sf_raw_e8m0"), .bytes = try store.read(arena, io, "shared_l1_sf_raw_e8m0") },
        }, .{ expected[0], expected[1] });
        // `shared_l2_w` is the raw input (unchanged by the transform).
        mismatched += try check(2, sharedL2, label, .{ "shared_l2_w", "shared_l2_sf" }, .{
            .{ .shape = try store.shape("shared_l2_w"), .bytes = expected[2] },
            .{ .shape = try store.shape("shared_l2_sf_raw_e8m0"), .bytes = try store.read(arena, io, "shared_l2_sf_raw_e8m0") },
        }, .{ expected[2], expected[3] });
    }
    return mismatched;
}

fn checkStore(dir: []const u8) !usize {
    const io = std.testing.io;
    var mismatched: usize = 0;
    var rank: usize = 0;
    while (true) : (rank += 1) {
        const path = try std.fmt.allocPrint(std.testing.allocator, "{s}/rank{d}.safetensors", .{ dir, rank });
        defer std.testing.allocator.free(path);
        var store = try Store.open(io, path) orelse break;
        defer store.deinit();
        mismatched += if (store.has("l1_w_raw")) try checkRandomRank(dir, rank, &store) else try checkRealRank(dir, rank, &store);
    }
    if (rank == 0) return error.NoRankFiles;
    const path = try std.fmt.allocPrint(std.testing.allocator, "{s}/shared.safetensors", .{dir});
    defer std.testing.allocator.free(path);
    if (try Store.open(io, path)) |shared| {
        var store = shared;
        defer store.deinit();
        mismatched += try checkShared(dir, &store);
    }
    return mismatched;
}

test "transforms match the reference weight stores byte for byte" {
    const dirs = std.c.getenv("MEGAMOE_WEIGHTS_STORES") orelse return error.SkipZigTest;
    var it = std.mem.splitScalar(u8, std.mem.span(dirs), ',');
    var failures: usize = 0;
    while (it.next()) |dir| {
        if (dir.len == 0) continue;
        const mismatched = checkStore(dir) catch |err| {
            std.debug.print("store {s}: {t}\n", .{ dir, err });
            failures += 1;
            continue;
        };
        std.debug.print("store {s}: {d} bytes differ\n", .{ dir, mismatched });
        if (mismatched != 0) failures += 1;
    }
    try std.testing.expectEqual(@as(usize, 0), failures);
}

// ==================== expert-sharded transforms ====================
//
// MEGAMOE_WEIGHTS_SHARDED=<real_* store dir> (two devices): both ranks' checkpoint experts as one
// tensor sharded by expert over the two devices (each device holds one rank's experts, like
// llmd's EP layout), outputs constrained to the same sharding. The optimized HLO must contain no
// collective, and the outputs must match the stores.

fn expertSharded(t: Tensor) Tensor {
    return t.withPartitioning(zml.testing.tpMesh(), .{ .expert = .model });
}

fn shardedRoutedL1(in: [4]Tensor) [2]Tensor {
    const out = routedL1(in);
    return .{ expertSharded(out[0]), expertSharded(out[1]) };
}

fn shardedRoutedL2(in: [2]Tensor) [2]Tensor {
    const out = routedL2(in);
    return .{ expertSharded(out[0]), expertSharded(out[1]) };
}

/// Collective ops in the optimized HLO modules XLA dumped to `dump_dir`.
fn countCollectives(dump_dir: []const u8) !usize {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var dir = try std.Io.Dir.cwd().openDir(io, dump_dir, .{ .iterate = true });
    defer dir.close(io);
    var optimized: usize = 0;
    var found: usize = 0;
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (std.mem.indexOf(u8, entry.name, "after_optimizations") == null) continue;
        optimized += 1;
        const bytes = try dir.readFileAlloc(io, entry.name, allocator, .limited(1 << 30));
        defer allocator.free(bytes);
        for ([_][]const u8{ "all-gather", "all-reduce", "all-to-all", "reduce-scatter", "collective-permute", "collective-broadcast" }) |op| {
            const count = std.mem.count(u8, bytes, op);
            if (count == 0) continue;
            std.debug.print("{s}: {d} x {s}\n", .{ entry.name, count, op });
            found += count;
        }
    }
    if (optimized == 0) return error.NoOptimizedHloDump;
    return found;
}

test "expert-sharded transforms: no communication, outputs match the stores" {
    const dir = std.c.getenv("MEGAMOE_WEIGHTS_SHARDED") orelse return error.SkipZigTest;
    const store_dir = std.mem.span(dir);
    const platform = zml.testing.env();
    if (platform.devices.len != 2) {
        std.debug.print("MEGAMOE_WEIGHTS_SHARDED needs two devices, got {d}\n", .{platform.devices.len});
        return error.TestUnexpectedResult;
    }
    const mesh = zml.testing.tpMesh();
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const io = std.testing.io;

    // Rank r's experts become shard r of one tensor.
    var inputs: [6]Input = undefined;
    var expected: [4][]const u8 = undefined;
    {
        var ranks: [2]RealExperts = undefined;
        for (&ranks, 0..) |*experts, r| {
            var store = (try Store.open(io, try std.fmt.allocPrint(arena, "{s}/rank{d}.safetensors", .{ store_dir, r }))) orelse return error.FileNotFound;
            defer store.deinit();
            experts.* = try .load(arena, store_dir, r, &store);
        }
        if (ranks[0].ids.len != ranks[1].ids.len) return error.UnevenExperts;
        for (&inputs, ranks[0].inputs, ranks[1].inputs) |*input, a, b| {
            const global = a.shape.setDim(0, a.shape.dim(0) + b.shape.dim(0));
            input.* = .{
                .shape = global.withTags(.{ .expert, .out, .in }).withPartitioning(mesh, .{ .expert = .model }),
                .bytes = try std.mem.concat(arena, u8, &.{ a.bytes, b.bytes }),
            };
        }
        for (&expected, ranks[0].expected, ranks[1].expected) |*e, a, b| e.* = try std.mem.concat(arena, u8, &.{ a, b });
    }

    const tmp = if (std.c.getenv("TEST_TMPDIR")) |t| std.mem.span(t) else "/tmp";
    const meshes = [_]*const zml.Mesh{mesh};
    const l1_dump = try std.fmt.allocPrint(arena, "{s}/megamoe_weights_sharded_l1", .{tmp});
    const l2_dump = try std.fmt.allocPrint(arena, "{s}/megamoe_weights_sharded_l2", .{tmp});
    var mismatched = try checkWith(4, shardedRoutedL1, "sharded routedL1", .{ "l1_w", "l1_sf" }, inputs[0..4].*, expected[0..2].*, .{ .meshes = &meshes, .xla_dump_to = l1_dump });
    mismatched += try checkWith(2, shardedRoutedL2, "sharded routedL2", .{ "l2_w", "l2_sf" }, inputs[4..6].*, expected[2..4].*, .{ .meshes = &meshes, .xla_dump_to = l2_dump });
    const collectives = try countCollectives(l1_dump) + try countCollectives(l2_dump);
    std.debug.print("sharded transforms: {d} bytes differ, {d} collectives in the optimized HLO\n", .{ mismatched, collectives });
    try std.testing.expectEqual(@as(usize, 0), mismatched);
    try std.testing.expectEqual(@as(usize, 0), collectives);
}

// ==================== end to end: transforms feeding forwardStaged ====================

const E2eInputs = struct {
    /// x, x_sf, topk_idx, topk_weights.
    staged: [4]Tensor,
    /// l1_w_raw, l1_sf_raw_e8m0, l2_w, l2_sf_raw_e8m0 and the shared expert's.
    raw: [8]Tensor,
    sm_locality_domains: Tensor,
    workspace: Tensor,
};

fn e2eCall(in: E2eInputs, problem: megamoe.Problem) [2]Tensor {
    const l1 = routedL1Joined(.{ in.raw[0], in.raw[1] });
    const l2 = routedL2(.{ in.raw[2], in.raw[3] });
    const shared = problem.num_shared_experts != 0;
    const shared_l1: ?[2]Tensor = if (shared) sharedL1Joined(.{ in.raw[4], in.raw[5] }) else null;
    const shared_l2: ?[2]Tensor = if (shared) sharedL2(.{ in.raw[6], in.raw[7] }) else null;
    const r = megamoe.forwardStaged(problem, .{
        .x = in.staged[0],
        .x_sf = in.staged[1],
        .topk_idx = in.staged[2],
        .topk_weights = in.staged[3],
    }, .{
        .l1_w = l1[0],
        .l1_sf = l1[1],
        .l2_w = l2[0],
        .l2_sf = l2[1],
        .shared_l1_w = if (shared_l1) |s| s[0] else null,
        .shared_l1_sf = if (shared_l1) |s| s[1] else null,
        .shared_l2_w = if (shared_l2) |s| s[0] else null,
        .shared_l2_sf = if (shared_l2) |s| s[1] else null,
    }, in.sm_locality_domains, .{ .operand = in.workspace });
    return .{ r.y, r.workspace.? };
}

fn runE2e(dir: []const u8) !usize {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var arena_state: std.heap.ArenaAllocator = .init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const Info = struct {
        num_ranks: u32,
        num_tokens: u32,
        num_experts: u32,
        num_topk: u32,
        hidden: u32,
        intermediate_hidden: u32,
        num_shared_experts: u32,
        num_max_tokens_per_rank_requested: u32,
        activation_clamp: f32,
        weights_file: []const u8,
        shared_weights_file: ?[]const u8 = null,
    };
    const json_bytes = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(arena, &.{ dir, "rank0.json" }), arena, .limited(1 << 24));
    const info = try std.json.parseFromSliceLeaky(Info, arena, json_bytes, .{ .ignore_unknown_fields = true });
    if (info.num_ranks != 1) return error.SingleRankDumpsOnly;
    const problem: megamoe.Problem = .{
        .num_ranks = 1,
        .num_experts = info.num_experts,
        .num_topk = info.num_topk,
        .hidden = info.hidden,
        .intermediate = info.intermediate_hidden,
        .num_shared_experts = info.num_shared_experts,
        .num_max_tokens_per_rank = info.num_max_tokens_per_rank_requested,
        .activation_clamp = info.activation_clamp,
    };

    var rank_store = (try Store.open(io, try std.fs.path.join(arena, &.{ dir, "rank0.safetensors" }))) orelse return error.FileNotFound;
    defer rank_store.deinit();
    var weights_store = (try Store.open(io, try std.fs.path.join(arena, &.{ dir, info.weights_file }))) orelse return error.FileNotFound;
    defer weights_store.deinit();
    const shared = problem.num_shared_experts != 0;
    var shared_store: ?Store = if (shared) try Store.open(io, try std.fs.path.join(arena, &.{ dir, info.shared_weights_file.? })) else null;
    defer if (shared_store) |*s| s.deinit();

    var buffers: std.ArrayList(zml.Buffer) = .empty;
    defer {
        for (buffers.items) |*b| b.deinit();
        buffers.deinit(allocator);
    }
    const Load = struct {
        fn f(list: *std.ArrayList(zml.Buffer), a: std.mem.Allocator, store: *const Store, name: []const u8) !zml.Buffer {
            const bytes = try store.read(a, std.testing.io, name);
            const buffer: zml.Buffer = try .fromBytes(std.testing.io, zml.testing.env(), try store.shape(name), bytes);
            try list.append(std.testing.allocator, buffer);
            return buffer;
        }
    };
    var staged: [4]zml.Buffer = undefined;
    for ([_][]const u8{ "x", "x_sf", "topk_idx", "topk_weights" }, &staged) |name, *b| b.* = try Load.f(&buffers, arena, &rank_store, name);
    const table = try Load.f(&buffers, arena, &rank_store, "sm_locality_domains");
    var raw: [8]zml.Buffer = undefined;
    for ([_][]const u8{ "l1_w_raw", "l1_sf_raw_e8m0", "l2_w", "l2_sf_raw_e8m0" }, raw[0..4]) |name, *b| b.* = try Load.f(&buffers, arena, &weights_store, name);
    if (shared_store) |*s| {
        for ([_][]const u8{ "shared_l1_w_raw", "shared_l1_sf_raw_e8m0", "shared_l2_w", "shared_l2_sf_raw_e8m0" }, raw[4..8]) |name, *b| b.* = try Load.f(&buffers, arena, s, name);
    } else {
        @memcpy(raw[4..8], raw[0..4]);
    }
    const y_ref = try rank_store.read(arena, io, "y");

    const num_sms: u32 = @intCast(table.shape().dim(0));
    const workspace_shape: zml.Shape = .init(.{@as(i64, @intCast(megamoe.workspaceBytes(problem, num_sms)))}, .u8);
    var inputs: E2eInputs = .{ .staged = undefined, .raw = undefined, .sm_locality_domains = .fromShape(table.shape()), .workspace = .fromShape(workspace_shape) };
    for (&inputs.staged, staged) |*t, b| t.* = .fromShape(b.shape());
    for (&inputs.raw, raw) |*t, b| t.* = .fromShape(b.shape());
    var exe = try platform.compileFn(allocator, io, e2eCall, .{ inputs, problem }, .{});
    defer exe.deinit();
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);

    const zeros = try arena.alloc(u8, workspace_shape.byteSize());
    @memset(zeros, 0);
    var workspace: zml.Buffer = try .fromBytes(io, platform, workspace_shape, zeros);
    defer workspace.deinit();
    var y: zml.Buffer = undefined;
    // The workspace input is donated: its output replaces the handle.
    runner.run(io, .{ staged, raw, table, workspace }, .{ &y, &workspace }, .{ .wait = true });
    defer y.deinit();
    const host = try y.toSliceAlloc(allocator, io);
    defer host.free(allocator);
    const label = try std.fmt.allocPrint(arena, "{s} transforms + forwardStaged", .{std.fs.path.basename(dir)});
    return countMismatches(label, "y", host.constData(), y_ref);
}

test "transformed raw weights through forwardStaged: y bit for bit" {
    const dirs = std.c.getenv("MEGAMOE_WEIGHTS_E2E") orelse return error.SkipZigTest;
    var it = std.mem.splitScalar(u8, std.mem.span(dirs), ',');
    var failures: usize = 0;
    while (it.next()) |dir| {
        if (dir.len == 0) continue;
        const mismatched = runE2e(dir) catch |err| {
            std.debug.print("dump {s}: {t}\n", .{ dir, err });
            failures += 1;
            continue;
        };
        if (mismatched != 0) failures += 1;
    }
    try std.testing.expectEqual(@as(usize, 0), failures);
}
