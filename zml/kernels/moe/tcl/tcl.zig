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
//!
//! Experts split across devices (expert parallel): each device runs its own
//! experts for the routes that reach them and an all-reduce sums the devices'
//! outputs. Up to `expert_stationary_tokens` tokens, expert stationary: a
//! device loop over the used experts, a few at a time, reads each once and
//! runs it for every token, scaled by the token's routing weight (zero for
//! the tokens it does not serve); more tokens take blockwise.
const std = @import("std");

const builder = @import("dsl/tcl/builder");

const stdx = @import("stdx");

const zml = @import("../../../zml.zig");
const Tensor = zml.Tensor;
const tcl = zml.kernel.tcl;

/// `input` is `{.t, .d}`, `topk_ids` and `topk_weights` `{.t, .topk}`, `gate_up` `{.expert, .dout = 2 * inter (gate then up), .d}`
/// and `down` `{.expert, .d, .dout = inter}`. Experts are BF16 or block-128 FP8 with BF16/F32 scales.
/// SwiGLU or GeGLU (tanh), routing weights applied after the down projection.
pub fn fusedExperts(input: Tensor, topk_ids: Tensor, topk_weights: Tensor, gate_up_linear: zml.nn.Linear, down_linear: zml.nn.Linear, opts: zml.moe.Options) Tensor {
    stdx.debug.assert(!opts.quantize_input and opts.routing_weight_placement == .after_down, "TCL MoE owns input precision and applies routing weights after the down projection", .{});
    stdx.debug.assert(opts.activation == .geglu_tanh or (opts.activation == .swiglu and std.meta.eql(opts.activation.swiglu, .{})), "TCL MoE supports GeGLU (tanh) and SwiGLU without limit, scale or bias", .{});
    stdx.debug.assert(gate_up_linear.bias == null and down_linear.bias == null, "TCL MoE takes experts without bias", .{});
    const gate_up = gate_up_linear.weight;
    const down = down_linear.weight;
    const quantized = gate_up_linear.quantization != null or down_linear.quantization != null;
    const scales: ?Scales = if (quantized) .{ .gate_up = fp8Scales(gate_up_linear), .down = fp8Scales(down_linear) } else null;
    stdx.debug.assert(input.dtype() == .bf16 and topk_ids.dtype() == .i32 and (quantized or (gate_up.dtype() == .bf16 and down.dtype() == .bf16)), "TCL MoE takes BF16 activations, BF16/block-128 FP8 experts and i32 ids", .{});
    const geglu = opts.activation == .geglu_tanh;
    const weights = topk_weights.convert(.f32);
    if (!gate_up.shape().partition(.expert).isSharded()) return runExperts(input, topk_ids, weights, gate_up, down, geglu, scales);
    return zml.ops.manualComputation(ExpertParallel.body, .{
        .input = input,
        .ids = topk_ids,
        .weights = weights,
        .gate_up = gate_up,
        .down = down,
        .geglu = geglu,
        .scales = scales,
    }, input.shape());
}

const Scales = struct { gate_up: Tensor, down: Tensor };

fn fp8Scales(linear: zml.nn.Linear) Tensor {
    const q = linear.quantization orelse std.debug.panic("TCL MoE requires both FP8 projections", .{});
    stdx.debug.assert(q.scheme == .fp8_block128 and linear.weight.dtype() == .f8e4m3fn and !q.swizzled_scales and q.global_scale == null and q.input_scale == null, "TCL MoE takes plain E4M3FN block-128 weights", .{});
    stdx.debug.assert(q.scales.dtype() == .bf16 or q.scales.dtype() == .f32, "TCL MoE FP8 scales must be BF16 or F32", .{});
    const w = linear.weight;
    const scale = q.scales;
    stdx.debug.assert(@mod(w.dim(.d), 128) == 0 and @mod(w.dim(.dout), 128) == 0 and scale.dim(.expert) == w.dim(.expert) and scale.dim(.d) * 128 == w.dim(.d) and scale.dim(.dout) * 128 == w.dim(.dout), "TCL MoE FP8 scales must cover 128x128 weight blocks", .{});
    return scale;
}

const ExpertParallel = struct {
    input: Tensor,
    ids: Tensor,
    weights: Tensor,
    gate_up: Tensor,
    down: Tensor,
    geglu: bool,
    scales: ?Scales,

    fn body(self: ExpertParallel, _: zml.Shape) Tensor {
        const experts = self.gate_up.dim(.expert);
        const start = zml.ops.partitionId().convert(.i32).scale(experts).convert(.i32);
        const ids = self.ids.sub(start.broad(self.ids.shape()));
        const mine = ids.cmp(.GE, Tensor.scalar(0, .i32).broad(ids.shape())).logical(.AND, ids.cmp(.LT, Tensor.scalar(experts, .i32).broad(ids.shape())));
        const local = localExpertsImpl(self.input, mine.select(ids, Tensor.scalar(-1, .i32)), mine.select(self.weights, Tensor.scalar(0, .f32)), self.gate_up, self.down, self.geglu, self.scales);
        return zml.ops.allReduce(local, Tensor.add);
    }
};

/// Tokens up to which a device runs its experts stationary.
pub const expert_stationary_tokens = 64;

/// One device's share of expert-parallel experts: `topk_ids` index the
/// device's own experts, -1 for routes to other devices' (weight zero).
pub fn localExperts(input: Tensor, topk_ids: Tensor, topk_weights: Tensor, gate_up: Tensor, down: Tensor, geglu: bool) Tensor {
    return localExpertsImpl(input, topk_ids, topk_weights, gate_up, down, geglu, null);
}

fn localExpertsImpl(input: Tensor, topk_ids: Tensor, topk_weights: Tensor, gate_up: Tensor, down: Tensor, geglu: bool, scales: ?Scales) Tensor {
    var cfg = config(input, topk_ids, gate_up, down, geglu);
    if (scales) |s| cfg.scale_dtype = scaleDtype(s);
    if (cfg.t > expert_stationary_tokens) {
        // Other devices' routes run this device's first expert, unweighted.
        const valid = topk_ids.cmp(.GE, Tensor.scalar(0, .i32).broad(topk_ids.shape()));
        return runExperts(input, valid.select(topk_ids, Tensor.scalar(0, .i32)), topk_weights, gate_up, down, geglu, scales);
    }
    // WORKAROUND (TCC 2026.5.0b0): a token count other than a power of two
    // panics its adder tree ("Fail to cover valid range", 5 tokens padded
    // to 8): pad with unrouted tokens.
    var padded = cfg;
    padded.t = @intCast(std.math.ceilPowerOfTwo(u64, @intCast(cfg.t)) catch unreachable);
    const pad = padded.t - cfg.t;
    const x = input.pad(0, .{ .t = Tensor.Pad{ .high = pad } }).reshape(.{ padded.t, cfg.d });
    const ids = topk_ids.pad(-1, .{ .t = Tensor.Pad{ .high = pad } }).reshape(.{ padded.t, cfg.k });
    const weights = topk_weights.pad(0, .{ .t = Tensor.Pad{ .high = pad } }).reshape(.{ padded.t, cfg.k });
    const y = if (scales) |s|
        ExpertStationaryFp8.call(.{ .x = x, .ids = ids, .w = weights, .gate_up = gate_up, .down = down, .gate_up_scale = s.gate_up, .down_scale = s.down }, .{ .y = x.shape() }, .{ .cfg = padded }).y
    else
        ExpertStationary.call(.{ .x = x, .ids = ids, .w = weights, .gate_up = gate_up, .down = down }, .{ .y = x.shape() }, .{ .cfg = padded }).y;
    return y.slice(0, .{ .end = cfg.t }).reshape(input.shape());
}

fn config(input: Tensor, topk_ids: Tensor, gate_up: Tensor, down: Tensor, geglu: bool) Cfg {
    const cfg: Cfg = .{
        .t = input.dim(.t),
        .k = topk_ids.dim(.topk),
        .e = gate_up.dim(.expert),
        .d = input.dim(.d),
        .i = down.dim(.dout),
        .geglu = geglu,
    };
    stdx.debug.assert(gate_up.dim(.dout) == 2 * cfg.i and down.dim(.d) == cfg.d and gate_up.dim(.d) == cfg.d, "TCL MoE: mismatched expert shapes", .{});
    return cfg;
}

fn runExperts(input: Tensor, topk_ids: Tensor, topk_weights: Tensor, gate_up: Tensor, down: Tensor, geglu: bool, scales: ?Scales) Tensor {
    var cfg = config(input, topk_ids, gate_up, down, geglu);
    const x = input.reshape(.{ cfg.t, cfg.d });
    const ids = topk_ids.reshape(.{ cfg.t, cfg.k });
    const weights = topk_weights.reshape(.{ cfg.t, cfg.k });
    const out_shape = x.shape();
    if (scales) |s| {
        cfg.scale_dtype = scaleDtype(s);
        const valid = ids.cmp(.GE, Tensor.scalar(0, .i32)).logical(.AND, ids.cmp(.LT, Tensor.scalar(cfg.e, .i32)));
        const inputs: ActivationStationaryFp8.Inputs = .{ .x = x, .ids = valid.select(ids, Tensor.scalar(0, .i32)), .w = valid.select(weights, Tensor.scalar(0, .f32)), .gate_up = gate_up, .down = down, .gate_up_scale = s.gate_up, .down_scale = s.down };
        const y = if (cfg.t * cfg.k <= cfg.e)
            ActivationStationaryFp8.call(inputs, .{ .y = out_shape }, .{ .cfg = cfg }).y
        else
            BlockwiseFp8.call(inputs, .{ .y = out_shape }, .{ .cfg = cfg }).y;
        return y.reshape(input.shape());
    }
    const y = if (cfg.t * cfg.k <= cfg.e)
        ActivationStationary.call(.{ .x = x, .ids = ids, .w = weights, .gate_up = gate_up, .down = down }, .{ .y = out_shape }, .{ .cfg = cfg }).y
    else
        Blockwise.call(.{ .x = x, .ids = ids, .w = weights, .gate_up = gate_up, .down = down }, .{ .y = out_shape }, .{ .cfg = cfg }).y;
    return y.reshape(input.shape());
}

fn scaleDtype(scales: Scales) tcl.DType {
    stdx.debug.assert(scales.gate_up.dtype() == scales.down.dtype(), "TCL MoE projection scales must have matching types", .{});
    return if (scales.gate_up.dtype() == .bf16) .bf16 else .f32;
}

pub const Cfg = struct { t: i64, k: i64, e: i64, d: i64, i: i64, geglu: bool = false, scale_dtype: tcl.DType = .bf16 };

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

fn args(b: *tcl.Builder, a: Axes, cfg: Cfg, comptime fp8: bool) tcl.FinishError!Args {
    const spec = .{
        .x = .{ .dtype = .bf16, .axes = &.{ a.T, a.D } },
        .ids = .{ .dtype = .i32, .axes = &.{ a.T, a.K } },
        .w = .{ .dtype = .f32, .axes = &.{ a.T, a.K } },
        .gate_up = .{ .dtype = if (fp8) .f8_e4 else .bf16, .axes = &.{ a.E, a.I2, a.D } },
        .down = .{ .dtype = if (fp8) .f8_e4 else .bf16, .axes = &.{ a.E, a.D, a.I } },
    };
    if (!fp8) {
        const t = try b.declareArgs(spec);
        return .{ .x = t.x, .ids = t.ids, .w = t.w, .gate_up = t.gate_up, .down = t.down };
    }
    const IG = b.axis("IG", @divExact(cfg.i, 128));
    const DG = b.axis("DG", @divExact(cfg.d, 128));
    const I2G = b.axis("I2G", 2 * @divExact(cfg.i, 128));
    const t = try b.declareArgs(.{
        .x = spec.x,
        .ids = spec.ids,
        .w = spec.w,
        .gate_up = spec.gate_up,
        .down = spec.down,
        .gate_up_scale = .{ .dtype = cfg.scale_dtype, .axes = &.{ a.E, I2G, DG } },
        .down_scale = .{ .dtype = cfg.scale_dtype, .axes = &.{ a.E, DG, IG } },
    });
    return .{ .x = t.x, .ids = t.ids, .w = t.w, .gate_up = t.gate_up, .down = t.down, .gate_up_scale = t.gate_up_scale, .down_scale = t.down_scale };
}

const Args = struct {
    x: builder.Tensor,
    ids: builder.Tensor,
    w: builder.Tensor,
    gate_up: builder.Tensor,
    down: builder.Tensor,
    gate_up_scale: ?builder.Tensor = null,
    down_scale: ?builder.Tensor = null,
};

const GatheredScales = struct { gate_up: ?builder.Tensor = null, down: ?builder.Tensor = null };

fn gatherScales(b: *tcl.Builder, a: Axes, in: Args, ids: builder.Tensor, batch: []const tcl.Axis) tcl.FinishError!GatheredScales {
    const gu = in.gate_up_scale orelse return .{};
    const IG = b.axis("IG", @divExact(a.I.size, 128));
    const DG = b.axis("DG", @divExact(a.D.size, 128));
    const gate_up = try b.reshape(gu, &.{ a.E, a.S2, IG, DG }, .{});
    var gu_axes: [8]tcl.Axis = undefined;
    var dn_axes: [8]tcl.Axis = undefined;
    @memcpy(gu_axes[0..batch.len], batch);
    @memcpy(dn_axes[0..batch.len], batch);
    @memcpy(gu_axes[batch.len..][0..3], &[_]tcl.Axis{ a.S2, IG, DG });
    @memcpy(dn_axes[batch.len..][0..2], &[_]tcl.Axis{ DG, IG });
    return .{
        .gate_up = try b.gather(gate_up, ids, a.E, gu_axes[0 .. batch.len + 3], .{}),
        .down = try b.gather(in.down_scale.?, ids, a.E, dn_axes[0 .. batch.len + 2], .{}),
    };
}

fn blockedAxes(axes: []const tcl.Axis, axis: tcl.Axis, group: tcl.Axis, lane: tcl.Axis, storage: *[16]tcl.Axis) []const tcl.Axis {
    var n: usize = 0;
    for (axes) |a| {
        if (std.mem.eql(u8, a.name, axis.name)) {
            storage[n] = group;
            n += 1;
            storage[n] = lane;
        } else storage[n] = a;
        n += 1;
    }
    return storage[0..n];
}

/// Block scales multiply each K-block partial before K-block reduction.
/// Only the requested experts are gathered; their bank remains FP8 in HBM.
fn project(b: *tcl.Builder, x: builder.Tensor, weight: builder.Tensor, scale: ?builder.Tensor, contracting: tcl.Axis, output: tcl.Axis, to: []const tcl.Axis, dtype: ?tcl.DType) tcl.FinishError!builder.Tensor {
    if (scale == null) {
        const op = b.tensorOperation(.{});
        return op.commit(op.contract(x, weight, to), .{ .dtype = dtype });
    }
    const CG = b.axis(if (std.mem.eql(u8, contracting.name, "D")) "DG" else "IG", @divExact(contracting.size, 128));
    const CL = b.axis(if (std.mem.eql(u8, contracting.name, "D")) "DB" else "IB", 128);
    const OG = b.axis(if (std.mem.eql(u8, output.name, "D")) "DG" else "IG", @divExact(output.size, 128));
    const OL = b.axis(if (std.mem.eql(u8, output.name, "D")) "DB" else "IB", 128);
    var x_axes: [16]tcl.Axis = undefined;
    var w_axes: [16]tcl.Axis = undefined;
    var w2_axes: [16]tcl.Axis = undefined;
    var out_axes: [16]tcl.Axis = undefined;
    var partial_axes: [16]tcl.Axis = undefined;
    const xb = try b.reshape(x, blockedAxes(x.axes, contracting, CG, CL, &x_axes), .{});
    const wb = try b.reshape(weight, blockedAxes(blockedAxes(weight.axes, output, OG, OL, &w_axes), contracting, CG, CL, &w2_axes), .{});
    const out = blockedAxes(to, output, OG, OL, &out_axes);
    @memcpy(partial_axes[0..out.len], out);
    partial_axes[out.len] = CG;

    // Furiosa's published FP8 recipe uses an additive epsilon, including zero blocks.
    var op = b.tensorOperation(.{});
    const qscale = try op.commit(op.fetch(xb, .{}).unary(.absf).reduce(&.{CL}, .maxf).divf(448.0).addf(1e-12), .{ .dtype = .f32 });
    op = b.tensorOperation(.{});
    // Native F32 reciprocal accuracy is qualified separately (within one ULP).
    const reciprocal = op.ve(.divf, .{ 1.0, op.fetch(qscale, .{}) });
    const quantized = try op.commit(reciprocal.mulf(op.fetch(xb, .{})), .{ .dtype = .f8_e4 });
    op = b.tensorOperation(.{});
    const scaled = op.contract(quantized, wb, partial_axes[0 .. out.len + 1]).mulf(op.fetch(qscale, .{})).mulf(op.fetch(scale.?, .{})).reduce(&.{CG}, .addf);
    return b.reshape(try op.commit(scaled, .{ .dtype = dtype, .axes = out }), to, .{});
}

/// `silu(gate) * up` or `gelu_tanh(gate) * up` of `h = [batch..., S2, I]`
/// (gate first), times `weights` (`[batch...]`) unless null.
fn gated(b: *tcl.Builder, a: Axes, h: builder.Tensor, batch: []const tcl.Axis, weights: ?builder.Tensor, geglu: bool, comptime fp8: bool) tcl.FinishError!builder.Tensor {
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
    // Interleaving requires one binary instruction; SwiGLU chains several.
    var op = b.tensorOperation(.{ .tactic = .elementwise });
    const g = op.fetch(gate, .{});
    // gelu_tanh(g) = g * sigmoid(2 * sqrt(2 / pi) * (g + 0.044715 * g^3)).
    const z = if (geglu) g.mulf(g).mulf(0.044715).addf(1.0).mulf(g).mulf(1.5957691216057308) else g;
    const activated = if (fp8) blk: {
        // WORKAROUND (TCC 2026.5.0b0): native fusion elides an internal
        // BF16 SiLU write, even through as_dram/as_logical. Make its RNE
        // rounding observable as integer bits before multiplying by up.
        // Remove once the native reproducer preserves the BF16 boundary.
        const bits = try op.commit(z.sigmoid().mulf(g).unary(.reinterpret_as_int), .{ .dtype = .i32 });
        op = b.tensorOperation(.{ .tactic = .elementwise });
        const v = op.fetch(bits, .{});
        const rounded = v.binary(.shr_logical, 16).binary(.bit_andi, 1).addi(32767).addi(v).binary(.bit_andi, -65536).unary(.reinterpret_as_float);
        const silu = try op.commit(rounded, .{ .dtype = .bf16 });
        op = b.tensorOperation(.{ .tactic = .elementwise });
        break :blk op.fetch(silu, .{});
    } else z.sigmoid().mulf(g);
    var v = activated.mulf(op.fetch(up, .{}));
    if (weights) |w| v = v.mulf(op.fetch(w, .{}));
    return op.commit(v, .{ .dtype = .bf16 });
}

/// Furiosa's compiler configuration for MoE kernels (`qwen3_moe/config.py`):
/// IO-bound tactics for activation stationary, compute-bound for blockwise.
/// Keep the compiler's default lowering mode and omit the adaptive-einsum and
/// half-VRF settings removed in TCC 2026.5.
fn configure(b: *tcl.Builder, blockwise: bool) void {
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
        .tactic_hint = .{ .ForLlmModelIOBound = 200 },
        .num_transaction_simulation_per_pe = 1024,
        .dma_preference = 0.8,
        .propagate_sparse_axis_from_op = "None",
    });
}

pub const ActivationStationary = activationStationaryKernel(false);
pub const ActivationStationaryFp8 = activationStationaryKernel(true);

fn activationStationaryKernel(comptime fp8: bool) type {
    return tcl.Kernel(Cfg, .{
        .name = if (fp8) "moe_activation_stationary_fp8" else "moe_activation_stationary",
        .inputs = if (fp8) &.{ "x", "ids", "w", "gate_up", "down", "gate_up_scale", "down_scale" } else &.{ "x", "ids", "w", "gate_up", "down" },
        .outputs = &.{"y"},
        .run = struct {
            fn run(b: *tcl.Builder, cfg: Cfg) tcl.FinishError!void {
                configure(b, false);
                const a: Axes = .init(b, cfg);
                const in = try args(b, a, cfg, fp8);
                const gate_up = try b.reshape(in.gate_up, &.{ a.E, a.S2, a.I, a.D }, .{});
                const gu = try b.gather(gate_up, in.ids, a.E, &.{ a.T, a.K, a.S2, a.I, a.D }, .{});
                const dn = try b.gather(in.down, in.ids, a.E, &.{ a.T, a.K, a.D, a.I }, .{});

                const scales = try gatherScales(b, a, in, in.ids, &.{ a.T, a.K });
                const h = try project(b, in.x, gu, scales.gate_up, a.D, a.I, &.{ a.T, a.K, a.S2, a.I }, if (fp8) .bf16 else null);
                const act = try gated(b, a, h, &.{ a.T, a.K }, if (fp8) null else in.w, cfg.geglu, fp8);
                if (fp8) {
                    const y = try project(b, act, dn, scales.down, a.I, a.D, &.{ a.T, a.K, a.D }, .f32);
                    const op = b.tensorOperation(.{});
                    b.ret(&.{try op.commit(op.fetch(y, .{}).mulf(op.fetch(in.w, .{})).reduce(&.{a.K}, .addf), .{ .dtype = .bf16 })});
                } else {
                    // The BF16 recipe weights activations and contracts experts together.
                    const op = b.tensorOperation(.{});
                    b.ret(&.{try op.commit(op.contract(act, dn, &.{ a.T, a.D }), .{ .dtype = .bf16 })});
                }
            }
        }.run,
    });
}

/// Rows per expert block: the next power of two of the routes per expert.
pub fn blockRows(cfg: Cfg) i64 {
    const per_expert = std.math.divCeil(i64, cfg.t * cfg.k, cfg.e) catch unreachable;
    return @intCast(std.math.ceilPowerOfTwo(u64, @intCast(@max(per_expert, 4))) catch unreachable);
}

const blocks_per_chunk = 32;

pub const Blockwise = blockwiseKernel(false);
pub const BlockwiseFp8 = blockwiseKernel(true);

fn blockwiseKernel(comptime fp8: bool) type {
    return tcl.Kernel(Cfg, .{
        .name = if (fp8) "moe_blockwise_fp8" else "moe_blockwise",
        .inputs = if (fp8) &.{ "x", "ids", "w", "gate_up", "down", "gate_up_scale", "down_scale" } else &.{ "x", "ids", "w", "gate_up", "down" },
        .outputs = &.{"y"},
        .run = struct {
            fn run(b: *tcl.Builder, cfg: Cfg) tcl.FinishError!void {
                configure(b, true);
                const a: Axes = .init(b, cfg);
                const in = try args(b, a, cfg, fp8);
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

                var loop = b.openFor(used, .{try b.scratchpad(if (fp8) .f32 else .bf16, &.{ Go, BGi, BB, D })});
                {
                    const chunk = try b.indexRead(rows, loop.iv);
                    const ids = try b.indexRead(chunk_experts, loop.iv);
                    const gu = try b.gather(gate_up, ids, E, &.{ BGi, a.S2, I, D }, .{});
                    const dn = try b.gather(in.down, ids, E, &.{ BGi, D, I }, .{});
                    const scales = try gatherScales(b, a, in, ids, &.{BGi});
                    const h = try project(b, chunk, gu, scales.gate_up, D, I, &.{ BGi, BB, a.S2, I }, if (fp8) .bf16 else null);
                    const act = try gated(b, a, h, &.{ BGi, BB }, null, cfg.geglu, fp8);
                    const y = try project(b, act, dn, scales.down, I, D, &.{ BGi, BB, D }, if (fp8) .f32 else .bf16);
                    try loop.yield(.{try b.indexWrite(loop.carried[0], loop.iv, y)});
                }
                const out = loop.results[0];

                const per_route = try b.gather(try b.reshape(out, &.{ GB, D }, .{}), slot, GB, &.{ T, K, D }, .{});
                op = b.tensorOperation(.{});
                b.ret(&.{try op.commit(op.fetch(per_route, .{ .typecast_to = .f32 }).mulf(op.fetch(in.w, .{})).reduce(&.{K}, .addf), .{ .dtype = .bf16 })});
            }
        }.run,
    });
}

/// Experts per `ExpertStationary` loop step: more as the routes, so the
/// experts they use, grow.
pub fn stationaryChunk(cfg: Cfg) i64 {
    const routes: u64 = @intCast(std.math.divCeil(i64, cfg.t * cfg.k, 16) catch unreachable);
    return @intCast(std.math.clamp(std.math.ceilPowerOfTwo(u64, routes) catch unreachable, 2, 16));
}

/// The experts that `ids` use (-1 for none), `stationaryChunk` at a time in a
/// device loop: each is read once and runs for every token, the activations
/// scaled by the token's routing weight for it (zero when unrouted); the
/// steps' outputs are summed.
pub const ExpertStationary = expertStationaryKernel(false);
pub const ExpertStationaryFp8 = expertStationaryKernel(true);

fn expertStationaryKernel(comptime fp8: bool) type {
    return tcl.Kernel(Cfg, .{
        .name = if (fp8) "moe_expert_stationary_fp8" else "moe_expert_stationary",
        .inputs = if (fp8) &.{ "x", "ids", "w", "gate_up", "down", "gate_up_scale", "down_scale" } else &.{ "x", "ids", "w", "gate_up", "down" },
        .outputs = &.{"y"},
        .run = struct {
            fn run(b: *tcl.Builder, cfg: Cfg) tcl.FinishError!void {
                configure(b, false);
                const a: Axes = .init(b, cfg);
                const in = try args(b, a, cfg, fp8);
                const chunk = stationaryChunk(cfg);
                const steps = std.math.divCeil(i64, @min(cfg.e, cfg.t * cfg.k), chunk) catch unreachable;
                const C = b.axis("C", chunk);
                const Go = b.axis("Go", steps);
                const J = b.axis("J", steps * chunk);
                const T, const K, const E, const D, const I = .{ a.T, a.K, a.E, a.D, a.I };

                // Each token's routing weight per expert, and the used experts.
                const experts = try b.arange(cfg.e, .i32, &.{E}, .{});
                var op = b.tensorOperation(.{ .tactic = .einsum_by_ve });
                const dist = try op.commit(op.fetch(in.ids, .{}).subi(op.fetch(experts, .{})), .{});
                op = b.tensorOperation(.{});
                const gate = try op.commit(op.where(op.fetch(dist, .{}), .eq, 0, op.fetch(in.w, .{ .broadcast_to = &.{ T, K, E } }), 0.0).reduce(&.{K}, .addf), .{});
                op = b.tensorOperation(.{});
                const onehot = try op.commit(op.where(dist, .eq, 0, 1, 0), .{});
                op = b.tensorOperation(.{});
                const counts = try op.commit(op.fetch(onehot, .{}).reduce(&.{ T, K }, .addi), .{});
                op = b.tensorOperation(.{});
                const used = try op.commit(op.where(counts, .gt, 0, 1, 0), .{});
                op = b.tensorOperation(.{});
                const position = try op.commit(op.fetch(used, .{}).cumsum(&.{E}), .{});
                // A used expert's rank among them, -1 for the others.
                op = b.tensorOperation(.{});
                const rank = try op.commit(op.fetch(position, .{}).muli(op.fetch(used, .{})).subi(1), .{});
                const trips = try b.symExpr(.div, try b.symExpr(.add, try b.reduceMaxI32(position), chunk - 1), chunk);

                // Slot j holds the used expert of rank j and its tokens' weights;
                // slots past the last hold expert 0 with no weight.
                const ranks = try b.arange(steps * chunk, .i32, &.{J}, .{});
                op = b.tensorOperation(.{ .tactic = .einsum_by_ve });
                const offset = try op.commit(op.fetch(rank, .{}).subi(op.fetch(ranks, .{})), .{});
                op = b.tensorOperation(.{});
                const slot_experts = try op.commit(op.where(op.fetch(offset, .{}), .eq, 0, op.fetch(experts, .{ .broadcast_to = &.{ E, J } }), 0).reduce(&.{E}, .addi), .{ .dtype = .i32 });
                // A reduction's condition is read plainly, so broadcast it first.
                op = b.tensorOperation(.{});
                const token_offset = try op.commit(op.fetch(offset, .{ .broadcast_to = &.{ T, E, J } }), .{});
                op = b.tensorOperation(.{});
                const slot_gate = try op.commit(op.where(op.fetch(token_offset, .{}), .eq, 0, op.fetch(gate, .{ .broadcast_to = &.{ T, E, J } }), 0.0).reduce(&.{E}, .addf), .{ .dtype = .f32, .axes = &.{ J, T } });
                const step_experts = try b.reshape(slot_experts, &.{ Go, C }, .{});
                const step_gate = try b.reshape(slot_gate, &.{ Go, C, T }, .{});
                const gate_up = try b.reshape(in.gate_up, &.{ E, a.S2, I, D }, .{});

                var loop = b.openFor(trips, .{try b.full(0, .f32, &.{ Go, T, D })});
                {
                    const ids = try b.indexRead(step_experts, loop.iv);
                    const weights = try b.indexRead(step_gate, loop.iv);
                    const gu = try b.gather(gate_up, ids, E, &.{ C, a.S2, I, D }, .{});
                    const dn = try b.gather(in.down, ids, E, &.{ C, D, I }, .{});
                    const scales = try gatherScales(b, a, in, ids, &.{C});
                    const h = try project(b, in.x, gu, scales.gate_up, D, I, &.{ C, T, a.S2, I }, if (fp8) .bf16 else null);
                    const act = try gated(b, a, h, &.{ C, T }, if (fp8) null else weights, cfg.geglu, fp8);
                    const y = if (fp8) blk: {
                        const per_expert = try project(b, act, dn, scales.down, I, D, &.{ C, T, D }, .f32);
                        op = b.tensorOperation(.{});
                        break :blk try op.commit(op.fetch(per_expert, .{}).mulf(op.fetch(weights, .{})).reduce(&.{C}, .addf), .{ .dtype = .f32 });
                    } else blk: {
                        op = b.tensorOperation(.{});
                        break :blk try op.commit(op.contract(act, dn, &.{ T, D }), .{ .dtype = .f32 });
                    };
                    try loop.yield(.{try b.indexWrite(loop.carried[0], loop.iv, y)});
                }
                op = b.tensorOperation(.{});
                b.ret(&.{try op.commit(op.fetch(loop.results[0], .{}).reduce(&.{Go}, .addf), .{ .dtype = .bf16 })});
            }
        }.run,
    });
}

test "tcl fused experts run on furiosa" {
    const platform = zml.testing.env();
    if (platform.target != .furiosa) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const bf16 = zml.floats.BFloat16;

    // 2 * 4 <= 8 experts is activation stationary, 16 tokens is blockwise;
    // one device's share of expert-parallel experts, some routes elsewhere
    // (-1), is expert stationary.
    const Case = struct { t: i64, local: bool = false, geglu: bool = false };
    for ([_]Case{ .{ .t = 4 }, .{ .t = 16 }, .{ .t = 1, .local = true }, .{ .t = 5, .local = true, .geglu = true }, .{ .t = 16, .local = true } }) |case| {
        const t = case.t;
        const k = 2;
        const e = 8;
        const d = 256;
        const i = 64;
        const x: Tensor = .init(.{ .t = t, .d = d }, .bf16);
        const ids: Tensor = .init(.{ .t = t, .topk = k }, .i32);
        const w: Tensor = .init(.{ .t = t, .topk = k }, .f32);
        const gu: Tensor = .init(.{ .expert = e, .dout = 2 * i, .d = d }, .bf16);
        const dn: Tensor = .init(.{ .expert = e, .d = d, .dout = i }, .bf16);
        const Mod = struct {
            pub fn forward(x_: Tensor, ids_: Tensor, w_: Tensor, gu_: Tensor, dn_: Tensor, local: bool, geglu: bool) Tensor {
                if (local) return localExperts(x_, ids_, w_, gu_, dn_, geglu);
                return fusedExperts(x_, ids_, w_, .init(gu_, null, .dout), .init(dn_, null, .dout), .{
                    .activation = if (geglu) .geglu_tanh else .{ .swiglu = .{} },
                    .quantize_input = false,
                    .routing_weight_placement = .after_down,
                });
            }
        };
        var exe = try zml.module.compile(allocator, io, Mod.forward, .{ x, ids, w, gu, dn, case.local, case.geglu }, platform, .{});
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
            // Another device serves the second route of odd tokens.
            hids[tok * k + j] = if (case.local and j == 1 and tok % 2 == 1) -1 else @intCast((tok * 3 + j * 5) % e);
            hw[tok * k + j] = if (hids[tok * k + j] < 0) 0 else if (j == 0) 0.7 else 0.3;
        };

        var bx: zml.Buffer = try .fromBytes(io, platform, x.shape(), std.mem.sliceAsBytes(hx));
        defer bx.deinit();
        var bids: zml.Buffer = try .fromBytes(io, platform, ids.shape(), std.mem.sliceAsBytes(hids));
        defer bids.deinit();
        var bw: zml.Buffer = try .fromBytes(io, platform, w.shape(), std.mem.sliceAsBytes(hw));
        defer bw.deinit();
        var bgu: zml.Buffer = try .fromBytes(io, platform, gu.shape(), std.mem.sliceAsBytes(hgu));
        defer bgu.deinit();
        var bdn: zml.Buffer = try .fromBytes(io, platform, dn.shape(), std.mem.sliceAsBytes(hdn));
        defer bdn.deinit();
        var result = try exe.eval(allocator, io, .{ bx, bids, bw, bgu, bdn });
        defer result.deinit();
        var host = try result.toSliceAlloc(allocator, io);
        defer host.free(allocator);
        const out = host.items(bf16);

        for (0..@intCast(t)) |tok| {
            var expected: [d]f32 = @splat(0);
            for (0..k) |j| {
                if (hids[tok * k + j] < 0) continue;
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
                    const z = if (case.geglu) 1.5957691216057308 * (g + 0.044715 * g * g * g) else g;
                    act[n] = g / (1 + @exp(-z)) * u;
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
