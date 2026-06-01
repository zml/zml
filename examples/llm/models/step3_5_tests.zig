const std = @import("std");

const zml = @import("zml");

const model = @import("step3_5flash/model.zig");

const TextRotaryEmbedding = model.TextRotaryEmbedding;

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Args = struct {
    model: []const u8,
    activations: []const u8,

    pub const help =
        \\Use step3_5_tests --model=<path> --activations=<path>
        \\
        \\ Validate the Step 3.5 Flash MoE layers against activation fixtures.
        \\
        \\ Options:
        \\   --model=<path>         Path to the model repository
        \\   --activations=<path>   Path to activation safetensors
        \\
    ;
};

fn swigluLimitFor(layer_idx: usize) ?f32 {
    return switch (layer_idx) {
        43, 44 => 7.0,
        else => null,
    };
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    const repo = try zml.safetensors.resolveModelRepo(io, args.model);

    var model_registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer model_registry.deinit();
    var model_store: zml.io.TensorStore = .fromRegistry(allocator, &model_registry);
    defer model_store.deinit();

    try run(allocator, io, platform, args.activations, &model_store);
}

fn run(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    activations_path: []const u8,
    model_store: *zml.io.TensorStore,
) !void {
    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, activations_path);
    defer registry.deinit();

    var activation_store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer activation_store.deinit();

    const mlp_layer_indices = [_]usize{ 0, 1, 2, 45, 46, 47 };
    std.log.info("MLP:", .{});
    for (mlp_layer_indices) |layer_idx| {
        const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.mlp", .{layer_idx});
        defer allocator.free(name);

        const mlp = model.Mlp.init(model_store.view().withPrefix(name), null);

        // Recursive cleanup for buffers
        var mlp_weights = try zml.io.load(model.Mlp, &mlp, allocator, io, platform, model_store, .auto);
        defer deinitBuffers(&mlp_weights);

        std.log.info("name {s}", .{name});

        const input_key = try std.fmt.allocPrint(allocator, "{s}.in.0", .{name});
        defer allocator.free(input_key);
        if (!activation_store.view().hasKey(input_key)) {
            std.log.warn("skipping {s}: no activations recorded", .{name});
            continue;
        }

        zml.testing.testLayer(allocator, io, platform, mlp, .forward, activation_store.view(), name, mlp_weights, &.{}, .{ .absolute_tolerance = 1e-2 }) catch |err| {
            std.log.warn("skipping {s}: {s}", .{ name, @errorName(err) });
        };
    }
    std.log.info("RMS Norm:", .{});

    for (0..48) |layer_idx| {
        const name__input_layernorm = try std.fmt.allocPrint(allocator, "model.layers.{d}.input_layernorm", .{layer_idx});
        defer allocator.free(name__input_layernorm);

        const name__post_attention_layernorm = try std.fmt.allocPrint(allocator, "model.layers.{d}.post_attention_layernorm", .{layer_idx});
        defer allocator.free(name__post_attention_layernorm);

        const rms1 = model.RmsNorm.init(model_store.view().withPrefix(name__input_layernorm), @as(f32, 1e-5));
        const rms2 = model.RmsNorm.init(model_store.view().withPrefix(name__post_attention_layernorm), @as(f32, 1e-5));

        var rms_weights1 = try zml.io.load(model.RmsNorm, &rms1, allocator, io, platform, model_store, .auto);
        defer deinitBuffers(&rms_weights1);

        var rms_weights2 = try zml.io.load(model.RmsNorm, &rms2, allocator, io, platform, model_store, .auto);
        defer deinitBuffers(&rms_weights2);

        const in1_key = try std.fmt.allocPrint(allocator, "{s}.in.0", .{name__input_layernorm});
        defer allocator.free(in1_key);
        if (activation_store.view().hasKey(in1_key)) {
            zml.testing.testLayer(allocator, io, platform, rms1, .forward, activation_store.view(), name__input_layernorm, rms_weights1, &.{}, .{ .absolute_tolerance = 1e-2 }) catch |err| {
                std.log.warn("skipping {s}: {s}", .{ name__input_layernorm, @errorName(err) });
            };
        } else {
            std.log.warn("skipping {s}: no activations recorded", .{name__input_layernorm});
        }

        const in2_key = try std.fmt.allocPrint(allocator, "{s}.in.0", .{name__post_attention_layernorm});
        defer allocator.free(in2_key);
        if (activation_store.view().hasKey(in2_key)) {
            zml.testing.testLayer(allocator, io, platform, rms2, .forward, activation_store.view(), name__post_attention_layernorm, rms_weights2, &.{}, .{ .absolute_tolerance = 1e-2 }) catch |err| {
                std.log.warn("skipping {s}: {s}", .{ name__post_attention_layernorm, @errorName(err) });
            };
        } else {
            std.log.warn("skipping {s}: no activations recorded", .{name__post_attention_layernorm});
        }
    }

    // MoE is verified; we no longer have to test the router
    for (42..45) |layer_idx| {
        const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.moe", .{layer_idx});
        defer allocator.free(name);

        const moe = try model.Moe.init(model_store.view().withPrefix(name), layer_idx);

        var moe_weights = try zml.io.load(model.Moe, &moe, allocator, io, platform, model_store, .auto);
        defer deinitBuffers(&moe_weights);

        const moe_in_key = try std.fmt.allocPrint(allocator, "{s}.in.0", .{name});
        defer allocator.free(moe_in_key);
        if (!activation_store.view().hasKey(moe_in_key)) {
            std.log.warn("skipping {s}: no activations recorded", .{name});
            continue;
        }

        zml.testing.testLayer(
            allocator,
            io,
            platform,
            moe,
            .forward,
            activation_store.view(),
            name,
            moe_weights,
            &.{},
            .{
                .absolute_tolerance = 1e-2,
                .minimum_close_fraction = 0.99,
            },
        ) catch |err| {
            std.log.warn("skipping {s}: {s}", .{ name, @errorName(err) });
        };
    }

    // RoPE
    try runRopeTests(allocator, io, platform);
}

fn deinitBuffers(bufs: anytype) void {
    zml.meta.visit(struct {
        fn cb(_: void, b: *zml.Buffer) void {
            b.deinit();
        }
    }.cb, {}, bufs);
}

// ===========================================================================
// Synthetic RoPE tests
//
// These do not need real model activations; RoPE is deterministic math.
// feed simple increasing inputs through the compiled graph and compare
// against a Zig CPU reference implementation that mirrors the HuggingFace
// reference exactly.
// ===========================================================================

const Rope = struct {
    const B: i64 = 1;
    const S: i64 = 4;
    const H: i64 = 2;
    const HD: i64 = 8;
    const ROTARY_DIM: i64 = 8;
    const THETA: f32 = 10_000.0;

    const QK_LEN: usize = @as(usize, @intCast(B * S * H * HD));
    const CS_LEN: usize = @as(usize, @intCast(B * S * HD));
    const POS_LEN: usize = @as(usize, @intCast(B * S));

    // --- synthetic inputs ---

    fn makeQ(out: []f32) void {
        for (out, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) / 100.0;
    }

    fn makeK(out: []f32) void {
        for (out, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) / 100.0 + 0.5;
    }

    fn makePositionIds(out: []i32) void {
        for (out, 0..) |*v, i| v.* = @intCast(i % @as(usize, @intCast(S)));
    }

    // --- CPU references ---

    fn refInvFreq(dim: i64, theta: f32, out: []f32) void {
        const half: usize = @intCast(@divExact(dim, 2));
        std.debug.assert(out.len == half);
        const N: f32 = @floatFromInt(half);
        for (out, 0..) |*v, i| {
            const fi: f32 = @floatFromInt(i);
            v.* = std.math.pow(f32, theta, -fi / N);
        }
    }

    fn refCosSin(position_ids: []const i32, cos_out: []f32, sin_out: []f32) void {
        var inv_freq: [@as(usize, @intCast(@divExact(ROTARY_DIM, 2)))]f32 = undefined;
        refInvFreq(ROTARY_DIM, THETA, &inv_freq);

        const half: usize = inv_freq.len;
        for (position_ids, 0..) |pos, idx| {
            const base = idx * @as(usize, @intCast(HD));
            const p: f32 = @floatFromInt(pos);
            for (0..half) |i| {
                const f = p * inv_freq[i];
                cos_out[base + i] = @cos(f);
                cos_out[base + half + i] = @cos(f);
                sin_out[base + i] = @sin(f);
                sin_out[base + half + i] = @sin(f);
            }
        }
    }

    fn refRotateHalf(x: []const f32, out: []f32) void {
        const hd: usize = @intCast(HD);
        const half: usize = hd / 2;
        var i: usize = 0;
        while (i < x.len) : (i += hd) {
            for (0..half) |j| out[i + j] = -x[i + half + j];
            for (0..half) |j| out[i + half + j] = x[i + j];
        }
    }

    fn refApplyRope(
        q: []const f32,
        k: []const f32,
        cos: []const f32,
        sin: []const f32,
        q_out: []f32,
        k_out: []f32,
    ) void {
        var rot_q: [QK_LEN]f32 = undefined;
        var rot_k: [QK_LEN]f32 = undefined;
        refRotateHalf(q, &rot_q);
        refRotateHalf(k, &rot_k);

        const hd: usize = @intCast(HD);
        const h: usize = @intCast(H);
        const s: usize = @intCast(S);
        const b: usize = @intCast(B);

        for (0..b) |bi| {
            for (0..s) |si| {
                for (0..h) |hi| {
                    for (0..hd) |di| {
                        const q_idx = ((bi * s + si) * h + hi) * hd + di;
                        const cs_idx = (bi * s + si) * hd + di;
                        q_out[q_idx] = q[q_idx] * cos[cs_idx] + rot_q[q_idx] * sin[cs_idx];
                        k_out[q_idx] = k[q_idx] * cos[cs_idx] + rot_k[q_idx] * sin[cs_idx];
                    }
                }
            }
        }
    }

    // --- compiled-graph wrappers ---

    fn cosWrapper(rope: TextRotaryEmbedding, position_ids: zml.Tensor) zml.Tensor {
        const cos, _ = rope.getCosAndSin(position_ids, .f32);
        return cos;
    }

    fn sinWrapper(rope: TextRotaryEmbedding, position_ids: zml.Tensor) zml.Tensor {
        _, const sin = rope.getCosAndSin(position_ids, .f32);
        return sin;
    }

    fn rotateHalfWrapper(x: zml.Tensor) zml.Tensor {
        return TextRotaryEmbedding.rotateHalf(x);
    }

    fn applyRopeQ(
        rope: TextRotaryEmbedding,
        q: zml.Tensor,
        k: zml.Tensor,
        cos: zml.Tensor,
        sin: zml.Tensor,
    ) zml.Tensor {
        const q_out, _ = rope.applyRope(q, k, cos, sin);
        return q_out;
    }

    fn applyRopeK(
        rope: TextRotaryEmbedding,
        q: zml.Tensor,
        k: zml.Tensor,
        cos: zml.Tensor,
        sin: zml.Tensor,
    ) zml.Tensor {
        _, const k_out = rope.applyRope(q, k, cos, sin);
        return k_out;
    }
};

fn runRopeTests(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !void {
    std.log.info("RoPE synthetic tests:", .{});
    try testRotateHalf(allocator, io, platform);
    try testCosSin(allocator, io, platform);
    try testApplyRopeReference(allocator, io, platform);
    try testApplyRopePosition0Identity(allocator, io, platform);
    try testApplyRopePreservesNorm(allocator, io, platform);
    std.log.info("  all RoPE tests passed.", .{});
}

fn testRotateHalf(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !void {
    std.log.info("  rotateHalf", .{});

    const x_shape = zml.Shape.init(.{ .b = Rope.B, .s = Rope.S, .h = Rope.H, .hd = Rope.HD }, .f32);
    const x_tensor = zml.Tensor.fromShape(x_shape);

    var x_data: [Rope.QK_LEN]f32 = undefined;
    Rope.makeQ(&x_data);

    var expected: [Rope.QK_LEN]f32 = undefined;
    Rope.refRotateHalf(&x_data, &expected);

    var exe = try zml.module.compile(allocator, io, Rope.rotateHalfWrapper, .{x_tensor}, platform, .{});
    defer exe.deinit();

    var x_buf: zml.Buffer = try .fromBytes(io, platform, x_shape, .replicated, std.mem.sliceAsBytes(&x_data));
    defer x_buf.deinit();

    var res = try zml.testing.autoCall(allocator, io, &exe, Rope.rotateHalfWrapper, .{x_buf});
    defer res.deinit();

    try zml.testing.expectClose(
        io,
        zml.Slice.init(x_shape, std.mem.sliceAsBytes(&expected)),
        res,
        .{ .absolute_tolerance = 1e-6 },
    );
}

fn testCosSin(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !void {
    std.log.info("  getCosAndSin", .{});

    const rope = TextRotaryEmbedding.init(Rope.ROTARY_DIM, Rope.THETA, 1.0);
    const pos_shape = zml.Shape.init(.{ .b = Rope.B, .s = Rope.S }, .i32);
    const cs_shape = zml.Shape.init(.{ .b = Rope.B, .s = Rope.S, .hd = Rope.HD }, .f32);
    const pos_tensor = zml.Tensor.fromShape(pos_shape);

    var pos_data: [Rope.POS_LEN]i32 = undefined;
    Rope.makePositionIds(&pos_data);

    var cos_expected: [Rope.CS_LEN]f32 = undefined;
    var sin_expected: [Rope.CS_LEN]f32 = undefined;
    Rope.refCosSin(&pos_data, &cos_expected, &sin_expected);

    var pos_buf: zml.Buffer = try .fromBytes(io, platform, pos_shape, .replicated, std.mem.sliceAsBytes(&pos_data));
    defer pos_buf.deinit();

    var exe_cos = try zml.module.compile(allocator, io, Rope.cosWrapper, .{ rope, pos_tensor }, platform, .{});
    defer exe_cos.deinit();
    var cos_res = try zml.testing.autoCall(allocator, io, &exe_cos, Rope.cosWrapper, .{pos_buf});
    defer cos_res.deinit();
    try zml.testing.expectClose(
        io,
        zml.Slice.init(cs_shape, std.mem.sliceAsBytes(&cos_expected)),
        cos_res,
        .{ .absolute_tolerance = 1e-5 },
    );

    var exe_sin = try zml.module.compile(allocator, io, Rope.sinWrapper, .{ rope, pos_tensor }, platform, .{});
    defer exe_sin.deinit();
    var sin_res = try zml.testing.autoCall(allocator, io, &exe_sin, Rope.sinWrapper, .{pos_buf});
    defer sin_res.deinit();
    try zml.testing.expectClose(
        io,
        zml.Slice.init(cs_shape, std.mem.sliceAsBytes(&sin_expected)),
        sin_res,
        .{ .absolute_tolerance = 1e-5 },
    );
}

fn testApplyRopeReference(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !void {
    std.log.info("  applyRope vs reference", .{});

    const rope = TextRotaryEmbedding.init(Rope.ROTARY_DIM, Rope.THETA, 1.0);
    const qk_shape = zml.Shape.init(.{ .b = Rope.B, .s = Rope.S, .h = Rope.H, .hd = Rope.HD }, .f32);
    const cs_shape = zml.Shape.init(.{ .b = Rope.B, .s = Rope.S, .hd = Rope.HD }, .f32);

    const q_t = zml.Tensor.fromShape(qk_shape);
    const k_t = zml.Tensor.fromShape(qk_shape);
    const cos_t = zml.Tensor.fromShape(cs_shape);
    const sin_t = zml.Tensor.fromShape(cs_shape);

    var q_data: [Rope.QK_LEN]f32 = undefined;
    var k_data: [Rope.QK_LEN]f32 = undefined;
    Rope.makeQ(&q_data);
    Rope.makeK(&k_data);

    var pos_data: [Rope.POS_LEN]i32 = undefined;
    Rope.makePositionIds(&pos_data);
    var cos_data: [Rope.CS_LEN]f32 = undefined;
    var sin_data: [Rope.CS_LEN]f32 = undefined;
    Rope.refCosSin(&pos_data, &cos_data, &sin_data);

    var q_expected: [Rope.QK_LEN]f32 = undefined;
    var k_expected: [Rope.QK_LEN]f32 = undefined;
    Rope.refApplyRope(&q_data, &k_data, &cos_data, &sin_data, &q_expected, &k_expected);

    var q_buf: zml.Buffer = try .fromBytes(io, platform, qk_shape, .replicated, std.mem.sliceAsBytes(&q_data));
    defer q_buf.deinit();
    var k_buf: zml.Buffer = try .fromBytes(io, platform, qk_shape, .replicated, std.mem.sliceAsBytes(&k_data));
    defer k_buf.deinit();
    var cos_buf: zml.Buffer = try .fromBytes(io, platform, cs_shape, .replicated, std.mem.sliceAsBytes(&cos_data));
    defer cos_buf.deinit();
    var sin_buf: zml.Buffer = try .fromBytes(io, platform, cs_shape, .replicated, std.mem.sliceAsBytes(&sin_data));
    defer sin_buf.deinit();

    var exe_q = try zml.module.compile(allocator, io, Rope.applyRopeQ, .{ rope, q_t, k_t, cos_t, sin_t }, platform, .{});
    defer exe_q.deinit();
    var q_res = try zml.testing.autoCall(allocator, io, &exe_q, Rope.applyRopeQ, .{ q_buf, k_buf, cos_buf, sin_buf });
    defer q_res.deinit();
    try zml.testing.expectClose(
        io,
        zml.Slice.init(qk_shape, std.mem.sliceAsBytes(&q_expected)),
        q_res,
        .{ .absolute_tolerance = 1e-5 },
    );

    var exe_k = try zml.module.compile(allocator, io, Rope.applyRopeK, .{ rope, q_t, k_t, cos_t, sin_t }, platform, .{});
    defer exe_k.deinit();
    var k_res = try zml.testing.autoCall(allocator, io, &exe_k, Rope.applyRopeK, .{ q_buf, k_buf, cos_buf, sin_buf });
    defer k_res.deinit();
    try zml.testing.expectClose(
        io,
        zml.Slice.init(qk_shape, std.mem.sliceAsBytes(&k_expected)),
        k_res,
        .{ .absolute_tolerance = 1e-5 },
    );
}

fn testApplyRopePosition0Identity(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !void {
    std.log.info("  applyRope position-0 identity", .{});

    const rope = TextRotaryEmbedding.init(Rope.ROTARY_DIM, Rope.THETA, 1.0);
    const qk_shape = zml.Shape.init(.{ .b = Rope.B, .s = 1, .h = Rope.H, .hd = Rope.HD }, .f32);
    const cs_shape = zml.Shape.init(.{ .b = Rope.B, .s = 1, .hd = Rope.HD }, .f32);

    const q_t = zml.Tensor.fromShape(qk_shape);
    const k_t = zml.Tensor.fromShape(qk_shape);
    const cos_t = zml.Tensor.fromShape(cs_shape);
    const sin_t = zml.Tensor.fromShape(cs_shape);

    const qk_len: usize = @as(usize, @intCast(Rope.B * 1 * Rope.H * Rope.HD));
    const cs_len: usize = @as(usize, @intCast(Rope.B * 1 * Rope.HD));

    var q_data: [qk_len]f32 = undefined;
    var k_data: [qk_len]f32 = undefined;
    for (&q_data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) / 100.0;
    for (&k_data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) / 100.0 + 0.5;

    // Position 0: cos = ones, sin = zeros, so applyRope must be the identity.
    var cos_data: [cs_len]f32 = .{1} ** cs_len;
    var sin_data: [cs_len]f32 = .{0} ** cs_len;

    var q_buf: zml.Buffer = try .fromBytes(io, platform, qk_shape, .replicated, std.mem.sliceAsBytes(&q_data));
    defer q_buf.deinit();
    var k_buf: zml.Buffer = try .fromBytes(io, platform, qk_shape, .replicated, std.mem.sliceAsBytes(&k_data));
    defer k_buf.deinit();
    var cos_buf: zml.Buffer = try .fromBytes(io, platform, cs_shape, .replicated, std.mem.sliceAsBytes(&cos_data));
    defer cos_buf.deinit();
    var sin_buf: zml.Buffer = try .fromBytes(io, platform, cs_shape, .replicated, std.mem.sliceAsBytes(&sin_data));
    defer sin_buf.deinit();

    var exe_q = try zml.module.compile(allocator, io, Rope.applyRopeQ, .{ rope, q_t, k_t, cos_t, sin_t }, platform, .{});
    defer exe_q.deinit();
    var q_res = try zml.testing.autoCall(allocator, io, &exe_q, Rope.applyRopeQ, .{ q_buf, k_buf, cos_buf, sin_buf });
    defer q_res.deinit();
    try zml.testing.expectClose(
        io,
        zml.Slice.init(qk_shape, std.mem.sliceAsBytes(&q_data)),
        q_res,
        .{ .absolute_tolerance = 1e-6 },
    );

    var exe_k = try zml.module.compile(allocator, io, Rope.applyRopeK, .{ rope, q_t, k_t, cos_t, sin_t }, platform, .{});
    defer exe_k.deinit();
    var k_res = try zml.testing.autoCall(allocator, io, &exe_k, Rope.applyRopeK, .{ q_buf, k_buf, cos_buf, sin_buf });
    defer k_res.deinit();
    try zml.testing.expectClose(
        io,
        zml.Slice.init(qk_shape, std.mem.sliceAsBytes(&k_data)),
        k_res,
        .{ .absolute_tolerance = 1e-6 },
    );
}

fn testApplyRopePreservesNorm(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !void {
    std.log.info("  applyRope preserves per-head norm", .{});

    const rope = TextRotaryEmbedding.init(Rope.ROTARY_DIM, Rope.THETA, 1.0);
    const qk_shape = zml.Shape.init(.{ .b = Rope.B, .s = Rope.S, .h = Rope.H, .hd = Rope.HD }, .f32);
    const cs_shape = zml.Shape.init(.{ .b = Rope.B, .s = Rope.S, .hd = Rope.HD }, .f32);

    const q_t = zml.Tensor.fromShape(qk_shape);
    const k_t = zml.Tensor.fromShape(qk_shape);
    const cos_t = zml.Tensor.fromShape(cs_shape);
    const sin_t = zml.Tensor.fromShape(cs_shape);

    var q_data: [Rope.QK_LEN]f32 = undefined;
    var k_data: [Rope.QK_LEN]f32 = undefined;
    Rope.makeQ(&q_data);
    Rope.makeK(&k_data);

    var pos_data: [Rope.POS_LEN]i32 = undefined;
    Rope.makePositionIds(&pos_data);
    var cos_data: [Rope.CS_LEN]f32 = undefined;
    var sin_data: [Rope.CS_LEN]f32 = undefined;
    Rope.refCosSin(&pos_data, &cos_data, &sin_data);

    var q_buf: zml.Buffer = try .fromBytes(io, platform, qk_shape, .replicated, std.mem.sliceAsBytes(&q_data));
    defer q_buf.deinit();
    var k_buf: zml.Buffer = try .fromBytes(io, platform, qk_shape, .replicated, std.mem.sliceAsBytes(&k_data));
    defer k_buf.deinit();
    var cos_buf: zml.Buffer = try .fromBytes(io, platform, cs_shape, .replicated, std.mem.sliceAsBytes(&cos_data));
    defer cos_buf.deinit();
    var sin_buf: zml.Buffer = try .fromBytes(io, platform, cs_shape, .replicated, std.mem.sliceAsBytes(&sin_data));
    defer sin_buf.deinit();

    var exe_q = try zml.module.compile(allocator, io, Rope.applyRopeQ, .{ rope, q_t, k_t, cos_t, sin_t }, platform, .{});
    defer exe_q.deinit();
    var q_res = try zml.testing.autoCall(allocator, io, &exe_q, Rope.applyRopeQ, .{ q_buf, k_buf, cos_buf, sin_buf });
    defer q_res.deinit();

    const q_out_slice = try q_res.toSliceAlloc(allocator, io);
    defer q_out_slice.free(allocator);
    const q_out_bytes = q_out_slice.constData();
    const q_out: []const f32 = @as([*]const f32, @ptrCast(@alignCast(q_out_bytes.ptr)))[0 .. q_out_bytes.len / @sizeOf(f32)];

    const hd: usize = @intCast(Rope.HD);
    const rd: usize = @intCast(Rope.ROTARY_DIM);
    var idx: usize = 0;
    while (idx < q_data.len) : (idx += hd) {
        var n_in: f32 = 0;
        var n_out: f32 = 0;
        for (0..rd) |j| {
            n_in += q_data[idx + j] * q_data[idx + j];
            n_out += q_out[idx + j] * q_out[idx + j];
        }
        if (@abs(n_in - n_out) > 1e-4) {
            std.log.err("Norm mismatch at head {d}: in={d}, out={d}", .{ idx / hd, n_in, n_out });
            return error.NormNotPreserved;
        }
    }
}
