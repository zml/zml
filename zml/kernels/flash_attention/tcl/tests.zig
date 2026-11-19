const std = @import("std");

const zml = @import("../../../zml.zig");
const BFloat16 = zml.floats.BFloat16;

test "tcl dense attention numerical tails and shapes on furiosa" {
    if (zml.testing.env().target != .furiosa) return error.SkipZigTest;
    try check(.{ .batch = 1, .queries = 64, .keys = 512, .heads = 4, .dim = 128, .constantValues = true });
    try check(.{ .batch = 1, .queries = 1, .keys = 257, .heads = 4, .dim = 64, .batched = false });
    try check(.{ .batch = 1, .queries = 65, .keys = 4099, .heads = 4, .dim = 32, .inputScale = 4 });
    try check(.{ .batch = 2, .queries = 577, .keys = 33, .heads = 4, .dim = 32 });
}

test "tcl dense attention wide heads on furiosa" {
    if (zml.testing.env().target != .furiosa) return error.SkipZigTest;
    try check(.{ .batch = 1, .queries = 257, .keys = 259, .heads = 56, .dim = 128 });
}

test "tcl dense attention single key block on furiosa" {
    if (zml.testing.env().target != .furiosa) return error.SkipZigTest;
    std.debug.print("TCL single key block: mesh devices={d}\n", .{zml.testing.tpMesh().numDevices()});
    try check(.{ .batch = 1, .queries = 9, .keys = 9, .heads = 56, .dim = 128 });
    // Multiple query blocks must share the same K/V without a gather.
    try check(.{ .batch = 1, .queries = 65, .keys = 9, .heads = 56, .dim = 128 });
}

const Case = struct {
    batch: usize,
    queries: usize,
    keys: usize,
    heads: usize,
    dim: usize,
    constantValues: bool = false,
    batched: bool = true,
    inputScale: f32 = 1,
};

fn forward(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, mesh: *const zml.Mesh) zml.Tensor {
    return zml.attention.attentionOpts(q, k, v, .scalar(0, .i32), .{ .flash_attention = .{ .tcl = {} } }, .{ .flash_attention = .{ .tcl = {} } }, .{ .causal = false }, mesh, .head(.model));
}

fn check(case: Case) !void {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    const mesh = zml.testing.tpMesh();
    var qShape: zml.Shape = .init(.{ .b = case.batch, .q = case.queries, .h = case.heads, .hd = case.dim }, .bf16);
    var kvShape: zml.Shape = .init(.{ .b = case.batch, .k = case.keys, .h = case.heads, .hd = case.dim }, .bf16);
    if (!case.batched) {
        qShape = qShape.drop(.b);
        kvShape = kvShape.drop(.b);
    }
    const q = try allocator.alloc(BFloat16, qShape.count());
    defer allocator.free(q);
    const k = try allocator.alloc(BFloat16, kvShape.count());
    defer allocator.free(k);
    const v = try allocator.alloc(BFloat16, kvShape.count());
    defer allocator.free(v);
    var prng: std.Random.DefaultPrng = .init(0);
    for (q) |*x| x.* = .fromF32((prng.random().float(f32) * 2 - 1) * case.inputScale);
    for (k) |*x| x.* = .fromF32((prng.random().float(f32) * 2 - 1) * case.inputScale);
    for (v) |*x| x.* = .fromF32(if (case.constantValues) 1 else prng.random().float(f32) * 2 - 1);

    const exe = try platform.compileFn(allocator, io, forward, .{
        zml.Tensor.fromShape(qShape), zml.Tensor.fromShape(kvShape), zml.Tensor.fromShape(kvShape), mesh,
    }, .{ .program_name = "tcl_dense_attention", .meshes = platform.meshes.values() });
    defer exe.deinit();
    var qBuf = try zml.Buffer.fromBytes(io, platform, qShape, std.mem.sliceAsBytes(q));
    defer qBuf.deinit();
    var kBuf = try zml.Buffer.fromBytes(io, platform, kvShape, std.mem.sliceAsBytes(k));
    defer kBuf.deinit();
    var vBuf = try zml.Buffer.fromBytes(io, platform, kvShape, std.mem.sliceAsBytes(v));
    defer vBuf.deinit();
    var output = try exe.eval(allocator, io, .{ qBuf, kBuf, vBuf });
    defer output.deinit();
    const host = try output.toSliceAlloc(allocator, io);
    defer host.free(allocator);
    try std.testing.expect(qShape.eqlWithTags(host.shape));

    // Independent f64 reference from the same quantized BF16 inputs.
    const scores = try allocator.alloc(f64, case.keys);
    defer allocator.free(scores);
    const expected = try allocator.alloc(f64, case.dim);
    defer allocator.free(expected);
    var maxError: f64 = 0;
    var failures: usize = 0;
    const scale = 1 / @sqrt(@as(f64, @floatFromInt(case.dim)));
    for (0..case.batch) |batch| {
        for (0..case.queries) |query| {
            for (0..case.heads) |head| {
                const qi = ((batch * case.queries + query) * case.heads + head) * case.dim;
                var maximum: f64 = -std.math.inf(f64);
                for (scores, 0..) |*score, key| {
                    const ki = ((batch * case.keys + key) * case.heads + head) * case.dim;
                    var dot: f64 = 0;
                    for (0..case.dim) |d| dot += @as(f64, q[qi + d].toF32()) * @as(f64, k[ki + d].toF32());
                    score.* = dot * scale;
                    maximum = @max(maximum, score.*);
                }
                var sum: f64 = 0;
                @memset(expected, 0);
                for (scores, 0..) |score, key| {
                    const weight = @exp(score - maximum);
                    sum += weight;
                    const vi = ((batch * case.keys + key) * case.heads + head) * case.dim;
                    for (expected, 0..) |*value, d| value.* += weight * @as(f64, v[vi + d].toF32());
                }
                for (expected, 0..) |value, d| {
                    const want = value / sum;
                    const got: f64 = host.items(BFloat16)[qi + d].toF32();
                    const err = @abs(got - want);
                    maxError = @max(maxError, err);
                    if (!std.math.isFinite(got) or err > 2e-3 + 1e-2 * @abs(want)) {
                        if (failures < 3) std.debug.print("mismatch b={d} q={d} h={d} d={d}: got={d} expected={d}\n", .{ batch, query, head, d, got, want });
                        failures += 1;
                    }
                }
            }
        }
    }
    std.debug.print("TCL dense attention b={d} q={d} k={d} h={d} d={d}: max_abs={d:.6} violations={d}\n", .{
        case.batch, case.queries, case.keys, case.heads, case.dim, maxError, failures,
    });
    try std.testing.expectEqual(@as(usize, 0), failures);
}
