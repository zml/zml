//! Furiosa RNGD fused experts in TCL, after Furiosa's production kernels.
//!
//! Few tokens (at most `experts / top_k`): activation stationary, each token
//! gathers the weights of its experts and the down projection contracts over
//! the experts and the intermediate axis at once.
//!
//! More tokens: blockwise. The kernel routes on the device: routes are placed
//! in expert-major blocks of `BB` rows (one-hot, cumulative sums), then a
//! device loop over the used chunks of `BGi` blocks gathers each block's
//! expert weights and runs both projections; outputs are gathered back per
//! route and weighted.
const std = @import("std");

const builder = @import("kernels/tcl/builder");

const stdx = @import("stdx");

const zml = @import("../zml.zig");
const Tensor = zml.Tensor;
const tcl = zml.kernel.tcl;

/// `input` is `{.b, .s, .d}`, `topk_ids` and `topk_weights` `{.b, .s,
/// .top_expert}`, `gate_up` `{.expert, .dout = 2 * inter (gate then up), .d}`
/// and `down` `{.expert, .d, .dout = inter}`, all bf16 but the ids (i32).
/// SwiGLU, routing weights applied after the down projection.
pub fn fusedExperts(input: Tensor, topk_ids: Tensor, topk_weights: Tensor, gate_up_linear: zml.nn.Linear, down_linear: zml.nn.Linear, opts: zml.moe.Options) Tensor {
    stdx.debug.assert(!opts.quantize_input and opts.routing_weight_placement == .after_down, "TCL MoE applies routing weights after the down projection, without FP8 input quantization", .{});
    stdx.debug.assert(opts.activation == .swiglu and std.meta.eql(opts.activation.swiglu, .{}), "TCL MoE supports SwiGLU without limit, scale or bias", .{});
    stdx.debug.assert(gate_up_linear.quantization == null and down_linear.quantization == null, "TCL MoE takes bf16 experts", .{});
    stdx.debug.assert(gate_up_linear.bias == null and down_linear.bias == null, "TCL MoE takes experts without bias", .{});
    const gate_up = gate_up_linear.weight;
    const down = down_linear.weight;
    stdx.debug.assert(input.dtype() == .bf16 and gate_up.dtype() == .bf16 and down.dtype() == .bf16 and topk_ids.dtype() == .i32, "TCL MoE takes bf16 activations and experts, i32 ids", .{});
    const t = input.dim(.b) * input.dim(.s);
    const k = topk_ids.dim(.top_expert);
    const cfg: Cfg = .{
        .t = t,
        .k = k,
        .e = gate_up.dim(.expert),
        .d = input.dim(.d),
        .i = down.dim(.dout),
    };
    stdx.debug.assert(gate_up.dim(.dout) == 2 * cfg.i and down.dim(.d) == cfg.d and gate_up.dim(.d) == cfg.d, "TCL MoE: mismatched expert shapes", .{});

    const x = input.reshape(.{ t, cfg.d });
    const ids = topk_ids.reshape(.{ t, k });
    const weights = topk_weights.convert(.f32).reshape(.{ t, k });
    const out_shape = x.shape();
    const y = if (cfg.t * cfg.k <= cfg.e)
        ActivationStationary.call(.{ .x = x, .ids = ids, .w = weights, .gate_up = gate_up, .down = down }, .{ .y = out_shape }, .{ .cfg = cfg }).y
    else
        Blockwise.call(.{ .x = x, .ids = ids, .w = weights, .gate_up = gate_up, .down = down }, .{ .y = out_shape }, .{ .cfg = cfg }).y;
    return y.reshape(input.shape());
}

pub const Cfg = struct { t: i64, k: i64, e: i64, d: i64, i: i64 };

const Axes = struct {
    T: tcl.Axis,
    K: tcl.Axis,
    E: tcl.Axis,
    D: tcl.Axis,
    I: tcl.Axis,
    I2: tcl.Axis,
    S2: tcl.Axis,
    S1: tcl.Axis,

    fn init(b: *tcl.Builder, cfg: Cfg) Axes {
        return .{
            .T = b.axis("T", cfg.t),
            .K = b.axis("K", cfg.k),
            .E = b.axis("E", cfg.e),
            .D = b.axis("D", cfg.d),
            .I = b.axis("I", cfg.i),
            .I2 = b.axis("I2", 2 * cfg.i),
            .S2 = b.axis("S2", 2),
            .S1 = b.axis("S1", 1),
        };
    }
};

fn args(b: *tcl.Builder, a: Axes) tcl.FinishError!Args {
    const t = try b.declareArgs(.{
        .x = .{ .dtype = .bf16, .axes = &.{ a.T, a.D } },
        .ids = .{ .dtype = .i32, .axes = &.{ a.T, a.K } },
        .w = .{ .dtype = .f32, .axes = &.{ a.T, a.K } },
        .gate_up = .{ .dtype = .bf16, .axes = &.{ a.E, a.I2, a.D } },
        .down = .{ .dtype = .bf16, .axes = &.{ a.E, a.D, a.I } },
    });
    return .{ .x = t.x, .ids = t.ids, .w = t.w, .gate_up = t.gate_up, .down = t.down };
}

const Args = struct { x: builder.Tensor, ids: builder.Tensor, w: builder.Tensor, gate_up: builder.Tensor, down: builder.Tensor };

/// `silu(gate) * up` of `h = [batch..., S2, I]` (gate first), times
/// `weights` (`[batch...]`) unless null.
fn gated(b: *tcl.Builder, a: Axes, h: builder.Tensor, batch: []const tcl.Axis, weights: ?builder.Tensor) tcl.FinishError!builder.Tensor {
    var sliced: [4]tcl.Axis = undefined;
    var flat: [3]tcl.Axis = undefined;
    @memcpy(sliced[0..batch.len], batch);
    @memcpy(flat[0..batch.len], batch);
    sliced[batch.len] = a.S1;
    sliced[batch.len + 1] = a.I;
    flat[batch.len] = a.I;
    const s = sliced[0 .. batch.len + 2];
    const f = flat[0 .. batch.len + 1];
    const gate = try b.reshape(try b.slice(h, a.S2, 0, s, .{}), f, .{});
    const up = try b.reshape(try b.slice(h, a.S2, 1, s, .{}), f, .{});
    // An `Interleaving` guess fails tcc's checks inside loops.
    const op = b.tensorOperation(.{ .tactic = .elementwise });
    const g = op.fetch(gate, .{});
    var v = g.sigmoid().mulf(g).mulf(op.fetch(up, .{}));
    if (weights) |w| v = v.mulf(op.fetch(w, .{}));
    return op.commit(v, .{ .dtype = .bf16 });
}

/// Furiosa's compiler configuration for MoE kernels (`qwen3_moe/config.py`):
/// IO-bound tactics for activation stationary, compute-bound for blockwise.
fn configure(b: *tcl.Builder, cfg: Cfg, blockwise: bool) void {
    if (blockwise) b.compilerConfig(.{
        .enable_einsum_fusion = true,
        .padding_policy = .{ .Small = 1.1 },
        .instruction_mem_budget = 0xB0000,
        .reshape_einsum_mode = .{ .Reshape = .{ .permute = true } },
        .tensor_unit_bridge_threshold_in_page = 12,
        .allow_external_operators = false,
        .enable_tactic_pruning = false,
        .scheduler_beam_search = true,
        .allow_reduce_by_ve_cluster_chip_reduce = false,
        .allow_reduce_by_ve_cluster_chip_reduce_base_population = false,
        .tactic_hint = .{ .ForLlmModelComputeBound = 200 },
        .apply_adaptive_einsum_by_pattern = true,
        .dma_preference = 1.2,
        .propagate_sparse_axis_from_op = "None",
    }) else b.compilerConfig(.{
        .enable_einsum_fusion = true,
        .padding_policy = .{ .Small = 1.1 },
        .instruction_mem_budget = 0xB0000,
        .reshape_einsum_mode = .{ .Reshape = .{ .permute = true } },
        .tensor_unit_bridge_threshold_in_page = 12,
        .allow_external_operators = false,
        .enable_tactic_pruning = false,
        .scheduler_beam_search = true,
        .allow_reduce_by_ve_cluster_chip_reduce = false,
        .allow_reduce_by_ve_cluster_chip_reduce_base_population = false,
        .lowering_mode = if (cfg.t < 8) "Optimal" else "Heuristic",
        .tactic_hint = .{ .ForLlmModelIOBound = 200 },
        .apply_adaptive_einsum_by_pattern = true,
        .num_transaction_simulation_per_pe = 1024,
        .dma_preference = 0.8,
        .enable_vrf_half_mode = true,
        .propagate_sparse_axis_from_op = "None",
    });
}

pub const ActivationStationary = tcl.Kernel(Cfg, .{
    .name = "moe_activation_stationary",
    .inputs = &.{ "x", "ids", "w", "gate_up", "down" },
    .outputs = &.{"y"},
    .run = struct {
        fn run(b: *tcl.Builder, cfg: Cfg) tcl.FinishError!void {
            configure(b, cfg, false);
            const a: Axes = .init(b, cfg);
            const in = try args(b, a);
            const gate_up = try b.reshape(in.gate_up, &.{ a.E, a.S2, a.I, a.D }, .{});
            const gu = try b.gather(gate_up, in.ids, a.E, &.{ a.T, a.K, a.S2, a.I, a.D }, .{});
            const dn = try b.gather(in.down, in.ids, a.E, &.{ a.T, a.K, a.D, a.I }, .{});

            var op = b.tensorOperation(.{});
            const h = try op.commit(op.contract(in.x, gu, &.{ a.T, a.K, a.S2, a.I }), .{});
            // Routing weights scale the activations, so the down projection
            // also sums over the experts.
            const act = try gated(b, a, h, &.{ a.T, a.K }, in.w);
            op = b.tensorOperation(.{});
            b.ret(&.{try op.commit(op.contract(act, dn, &.{ a.T, a.D }), .{ .dtype = .bf16 })});
        }
    }.run,
});

/// Rows per expert block: the next power of two of the routes per expert.
pub fn blockRows(cfg: Cfg) i64 {
    const per_expert = std.math.divCeil(i64, cfg.t * cfg.k, cfg.e) catch unreachable;
    return @intCast(std.math.ceilPowerOfTwo(u64, @intCast(@max(per_expert, 4))) catch unreachable);
}

const blocks_per_chunk = 32;

pub const Blockwise = tcl.Kernel(Cfg, .{
    .name = "moe_blockwise",
    .inputs = &.{ "x", "ids", "w", "gate_up", "down" },
    .outputs = &.{"y"},
    .run = struct {
        fn run(b: *tcl.Builder, cfg: Cfg) tcl.FinishError!void {
            configure(b, cfg, true);
            const a: Axes = .init(b, cfg);
            const in = try args(b, a);
            const bb = blockRows(cfg);
            // One block per expert at least, then enough for every route.
            const max_blocks = std.math.divCeil(i64, @max(cfg.t * cfg.k - cfg.e, 0), bb) catch unreachable;
            const chunks = std.math.divCeil(i64, max_blocks + cfg.e, blocks_per_chunk) catch unreachable;
            const BB = b.axis("BB", bb);
            const BGi = b.axis("BGi", blocks_per_chunk);
            const Go = b.axis("Go", chunks);
            const BG = b.axis("BG", chunks * blocks_per_chunk);
            const GB = b.axis("GB", chunks * blocks_per_chunk * bb);
            const T, const K, const E, const D, const I = .{ a.T, a.K, a.E, a.D, a.I };

            // Routing: route (t, k) goes to row `slot` of block `start[e] + rank`.
            const experts = try b.arange(cfg.e, .i32, &.{E}, .{});
            var op = b.tensorOperation(.{ .tactic = .einsum_by_ve });
            const dist = try op.commit(op.fetch(in.ids, .{}).subi(op.fetch(experts, .{})), .{});
            op = b.tensorOperation(.{});
            const onehot = try op.commit(op.where(dist, .eq, 0, 1, 0), .{});
            op = b.tensorOperation(.{});
            const rank = try op.commit(op.fetch(onehot, .{}).cumsum(&.{ T, K }), .{});
            op = b.tensorOperation(.{});
            const log2_bb: i64 = std.math.log2_int(u64, @intCast(bb));
            const blocks = try op.commit(op.fetch(onehot, .{}).reduce(&.{ T, K }, .addi).addi(bb - 1).binary(.shr_arith, log2_bb), .{});
            op = b.tensorOperation(.{});
            const block_end = try op.commit(op.fetch(blocks, .{}).cumsum(&.{E}), .{});
            op = b.tensorOperation(.{});
            const start_rows = try op.commit(op.fetch(block_end, .{}).subi(op.fetch(blocks, .{})).muli(bb), .{});
            op = b.tensorOperation(.{});
            const slot = try op.commit(op.fetch(rank, .{}).addi(op.fetch(start_rows, .{})).muli(op.fetch(onehot, .{})).reduce(&.{E}, .addi).subi(1), .{});
            // The expert of block g: how many experts end at or before g.
            const block_ids = try b.arange(chunks * blocks_per_chunk, .i32, &.{BG}, .{});
            op = b.tensorOperation(.{ .tactic = .einsum_by_ve });
            const ends = try op.commit(op.fetch(block_end, .{}).subi(op.fetch(block_ids, .{})), .{});
            op = b.tensorOperation(.{});
            const ended = try op.commit(op.where(ends, .le, 0, 1, 0), .{});
            op = b.tensorOperation(.{});
            // Padding blocks past the last expert reuse the last one.
            const block_expert = try op.commit(op.fetch(ended, .{}).reduce(&.{E}, .addi).binary(.mini, cfg.e - 1), .{});
            const used = try b.symExpr(.div, try b.symExpr(.add, try b.reduceMaxI32(block_end), blocks_per_chunk - 1), blocks_per_chunk);

            // Rows of unused slots are never read back.
            const rows = try b.reshape(try b.scatter(in.x, slot, GB, &.{ GB, D }, .{}), &.{ Go, BGi, BB, D }, .{});
            const chunk_experts = try b.reshape(block_expert, &.{ Go, BGi }, .{});
            const gate_up = try b.reshape(in.gate_up, &.{ E, a.S2, I, D }, .{});

            var loop = b.openFor(used, .{try b.scratchpad(.bf16, &.{ Go, BGi, BB, D })});
            const chunk = try b.indexRead(rows, loop.iv);
            const ids = try b.indexRead(chunk_experts, loop.iv);
            const gu = try b.gather(gate_up, ids, E, &.{ BGi, a.S2, I, D }, .{});
            const dn = try b.gather(in.down, ids, E, &.{ BGi, D, I }, .{});
            op = b.tensorOperation(.{});
            const h = try op.commit(op.contract(chunk, gu, &.{ BGi, BB, a.S2, I }), .{});
            const act = try gated(b, a, h, &.{ BGi, BB }, null);
            op = b.tensorOperation(.{});
            const y = try op.commit(op.contract(act, dn, &.{ BGi, BB, D }), .{ .dtype = .bf16 });
            try loop.yield(.{try b.indexWrite(loop.carried[0], loop.iv, y)});
            const out = loop.results[0];

            const per_route = try b.gather(try b.reshape(out, &.{ GB, D }, .{}), slot, GB, &.{ T, K, D }, .{});
            op = b.tensorOperation(.{});
            b.ret(&.{try op.commit(op.fetch(per_route, .{ .typecast_to = .f32 }).mulf(op.fetch(in.w, .{})).reduce(&.{K}, .addf), .{ .dtype = .bf16 })});
        }
    }.run,
});

test "tcl fused experts run on furiosa" {
    const platform = zml.testing.env();
    if (platform.target != .furiosa) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const bf16 = zml.floats.BFloat16;

    // 2 * 4 <= 8 experts is activation stationary, 16 tokens is blockwise.
    for ([_]i64{ 4, 16 }) |t| {
        const k = 2;
        const e = 8;
        const d = 256;
        const i = 64;
        const x: Tensor = .init(.{ .b = 1, .s = t, .d = d }, .bf16);
        const ids: Tensor = .init(.{ .b = 1, .s = t, .top_expert = k }, .i32);
        const w: Tensor = .init(.{ .b = 1, .s = t, .top_expert = k }, .f32);
        const gu: Tensor = .init(.{ .expert = e, .dout = 2 * i, .d = d }, .bf16);
        const dn: Tensor = .init(.{ .expert = e, .d = d, .dout = i }, .bf16);
        const Mod = struct {
            pub fn forward(x_: Tensor, ids_: Tensor, w_: Tensor, gu_: Tensor, dn_: Tensor) Tensor {
                return fusedExperts(x_, ids_, w_, .init(gu_, null, .dout), .init(dn_, null, .dout), .{
                    .activation = .{ .swiglu = .{} },
                    .quantize_input = false,
                    .routing_weight_placement = .after_down,
                });
            }
        };
        var exe = try zml.module.compile(allocator, io, Mod.forward, .{ x, ids, w, gu, dn }, platform, .{});
        defer exe.deinit();

        var prng: std.Random.DefaultPrng = .init(@intCast(t));
        const r = prng.random();
        const hx = try allocator.alloc(bf16, @intCast(t * d));
        defer allocator.free(hx);
        const hgu = try allocator.alloc(bf16, e * 2 * i * d);
        defer allocator.free(hgu);
        const hdn = try allocator.alloc(bf16, e * d * i);
        defer allocator.free(hdn);
        for (hx) |*v| v.* = .fromF32(r.floatNorm(f32) * 0.5);
        for (hgu) |*v| v.* = .fromF32(r.floatNorm(f32) / 16);
        for (hdn) |*v| v.* = .fromF32(r.floatNorm(f32) / 8);
        const hids = try allocator.alloc(i32, @intCast(t * k));
        defer allocator.free(hids);
        const hw = try allocator.alloc(f32, @intCast(t * k));
        defer allocator.free(hw);
        for (0..@intCast(t)) |tok| for (0..k) |j| {
            hids[tok * k + j] = @intCast((tok * 3 + j * 5) % e);
            hw[tok * k + j] = if (j == 0) 0.7 else 0.3;
        };

        var bx: zml.Buffer = try .fromBytes(io, platform, x.shape(), .replicated, std.mem.sliceAsBytes(hx));
        defer bx.deinit();
        var bids: zml.Buffer = try .fromBytes(io, platform, ids.shape(), .replicated, std.mem.sliceAsBytes(hids));
        defer bids.deinit();
        var bw: zml.Buffer = try .fromBytes(io, platform, w.shape(), .replicated, std.mem.sliceAsBytes(hw));
        defer bw.deinit();
        var bgu: zml.Buffer = try .fromBytes(io, platform, gu.shape(), .replicated, std.mem.sliceAsBytes(hgu));
        defer bgu.deinit();
        var bdn: zml.Buffer = try .fromBytes(io, platform, dn.shape(), .replicated, std.mem.sliceAsBytes(hdn));
        defer bdn.deinit();
        var result = try exe.eval(allocator, io, .{ bx, bids, bw, bgu, bdn });
        defer result.deinit();
        var host = try result.toSliceAlloc(allocator, io);
        defer host.free(allocator);
        const out = host.items(bf16);

        for (0..@intCast(t)) |tok| {
            var expected: [d]f32 = @splat(0);
            for (0..k) |j| {
                const ex: usize = @intCast(hids[tok * k + j]);
                var act: [i]f32 = undefined;
                for (0..i) |n| {
                    var g: f32 = 0;
                    var u: f32 = 0;
                    for (0..d) |m| {
                        const xv = hx[tok * d + m].toF32();
                        g += xv * hgu[(ex * 2 * i + n) * d + m].toF32();
                        u += xv * hgu[(ex * 2 * i + i + n) * d + m].toF32();
                    }
                    act[n] = g / (1 + @exp(-g)) * u;
                }
                for (0..d) |m| {
                    var y: f32 = 0;
                    for (0..i) |n| y += act[n] * hdn[(ex * d + m) * i + n].toF32();
                    expected[m] += hw[tok * k + j] * y;
                }
            }
            for (expected, out[tok * d ..][0..d]) |want, got| try std.testing.expectApproxEqAbs(want, got.toF32(), 2e-2 + 2e-2 * @abs(want));
        }
    }
}
