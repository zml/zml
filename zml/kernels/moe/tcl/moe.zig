//! TCL kernel schedules for Furiosa RNGD fused experts.
const std = @import("std");

const builder = @import("dsl/tcl/builder");

const zml = @import("zml/core");
const tcl = zml.kernel.tcl;
const ActivationKind = @import("zml/kernels/moe/api").ActivationKind;

pub const Format = enum { bf16, fp8_block128 };

/// Dispatch in `tcl.zig` uses the first matching schedule.
///
/// - `.small_batch` for FP8 up to 64 tokens before padding. Run each selected
///   expert on the whole batch to reuse its weights, with zero routing weight
///   for unrouted pairs.
/// - `.per_route` when `num_tokens * top_k <= local_num_experts`. Gather
///   weights for each route to avoid the overhead of grouping routes.
/// - `.blockwise` otherwise. Pack routed tokens into blocks by expert to
///   reuse weights within each block, then gather results back by route.
pub const Schedule = enum { per_route, blockwise, small_batch };

pub const Cfg = struct {
    num_tokens: i64,
    top_k: i64,
    local_num_experts: i64,
    hidden_size: i64,
    expert_intermediate_size: i64,
    weight_scale_dtype: tcl.DType = .bf16,
    activation: ActivationKind = .swiglu,
};

const base_inputs: []const [:0]const u8 = &.{ "x", "ids", "w", "gate_up", "down" };
const scale_inputs: []const [:0]const u8 = &.{ "gate_up_scale", "down_scale" };

pub fn Kernel(comptime format: Format, comptime schedule: Schedule) type {
    if (schedule == .small_batch and format != .fp8_block128) {
        @compileError("TCL MoE FP8 small-batch schedule requires block-128 FP8");
    }
    return tcl.Kernel(Cfg, .{
        .name = "moe_" ++ @tagName(schedule) ++ switch (format) {
            .bf16 => "",
            .fp8_block128 => "_fp8",
        },
        .inputs = switch (format) {
            .bf16 => base_inputs,
            .fp8_block128 => base_inputs ++ scale_inputs,
        },
        .outputs = &.{"y"},
        .run = struct {
            fn run(b: *tcl.Builder, cfg: Cfg) tcl.FinishError!void {
                configure(b, schedule);
                const a: Axes = .init(b, cfg);
                const in = try args(b, a, cfg, format);
                switch (schedule) {
                    .per_route => try perRoute(b, a, in, cfg.activation, format),
                    .blockwise => try blockwise(b, a, in, cfg, format),
                    .small_batch => try fp8SmallBatch(b, a, in, cfg),
                }
            }
        }.run,
    });
}

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
            .T = b.axis("T", cfg.num_tokens),
            .K = b.axis("K", cfg.top_k),
            .E = b.axis("E", cfg.local_num_experts),
            .D = b.axis("D", cfg.hidden_size),
            .I = b.axis("I", cfg.expert_intermediate_size),
            .I2 = b.axis("I2", 2 * cfg.expert_intermediate_size),
            .S2 = b.axis("S2", 2),
            .S1 = b.axis("S1", 1),
        };
    }
};

fn args(b: *tcl.Builder, a: Axes, cfg: Cfg, comptime format: Format) tcl.FinishError!Args {
    const weight_dtype: tcl.DType = switch (format) {
        .bf16 => .bf16,
        .fp8_block128 => .f8_e4,
    };
    const spec = .{
        .x = .{ .dtype = .bf16, .axes = &.{ a.T, a.D } },
        .ids = .{ .dtype = .i32, .axes = &.{ a.T, a.K } },
        .w = .{ .dtype = .f32, .axes = &.{ a.T, a.K } },
        .gate_up = .{ .dtype = weight_dtype, .axes = &.{ a.E, a.I2, a.D } },
        .down = .{ .dtype = weight_dtype, .axes = &.{ a.E, a.D, a.I } },
    };
    if (format == .bf16) {
        const t = try b.declareArgs(spec);
        return .{ .x = t.x, .ids = t.ids, .w = t.w, .gate_up = t.gate_up, .down = t.down };
    }
    const IG = b.axis("IG", @divExact(cfg.expert_intermediate_size, 128));
    const DG = b.axis("DG", @divExact(cfg.hidden_size, 128));
    const I2G = b.axis("I2G", 2 * @divExact(cfg.expert_intermediate_size, 128));
    const t = try b.declareArgs(.{
        .x = spec.x,
        .ids = spec.ids,
        .w = spec.w,
        .gate_up = spec.gate_up,
        .down = spec.down,
        .gate_up_scale = .{ .dtype = cfg.weight_scale_dtype, .axes = &.{ a.E, I2G, DG } },
        .down_scale = .{ .dtype = cfg.weight_scale_dtype, .axes = &.{ a.E, DG, IG } },
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

fn gatherScales(b: *tcl.Builder, a: Axes, in: Args, ids: builder.Tensor, batch: []const tcl.Axis, comptime format: Format) tcl.FinishError!GatheredScales {
    if (format == .bf16) return .{};
    const IG = b.axis("IG", @divExact(a.I.size, 128));
    const DG = b.axis("DG", @divExact(a.D.size, 128));
    const gate_up = try b.reshape(in.gate_up_scale.?, &.{ a.E, a.S2, IG, DG }, .{});
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
fn project(b: *tcl.Builder, x: builder.Tensor, weight: builder.Tensor, scale: ?builder.Tensor, contracting: tcl.Axis, output: tcl.Axis, to: []const tcl.Axis, dtype: ?tcl.DType, comptime format: Format) tcl.FinishError!builder.Tensor {
    if (format == .bf16) {
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
fn gated(b: *tcl.Builder, a: Axes, h: builder.Tensor, batch: []const tcl.Axis, weights: ?builder.Tensor, activation: ActivationKind, comptime format: Format) tcl.FinishError!builder.Tensor {
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
    // The builder can infer the TCL `Interleaving` tactic for two inputs with
    // matching axes, such as `gate` and `up`. That tactic allows exactly one
    // binary instruction, so force `.elementwise` for this activation chain.
    var op = b.tensorOperation(.{ .tactic = .elementwise });
    const g = op.fetch(gate, .{});
    const activated = if (format == .fp8_block128) blk: {
        if (activation != .swiglu) std.debug.panic("TCL MoE block-128 FP8 requires SwiGLU, got {s}", .{@tagName(activation)});

        // WORKAROUND (TCC 2026.5.0b0): native fusion elides an internal
        // BF16 SiLU write, even through as_dram/as_logical. Make its RNE
        // rounding observable as integer bits before multiplying by up.
        // Remove once the native reproducer preserves the BF16 boundary.
        const bits = try op.commit(g.sigmoid().mulf(g).unary(.reinterpret_as_int), .{ .dtype = .i32 });
        op = b.tensorOperation(.{ .tactic = .elementwise });
        const v = op.fetch(bits, .{});
        const rounded = v.binary(.shr_logical, 16).binary(.bit_andi, 1).addi(32767).addi(v).binary(.bit_andi, -65536).unary(.reinterpret_as_float);
        const silu = try op.commit(rounded, .{ .dtype = .bf16 });
        op = b.tensorOperation(.{ .tactic = .elementwise });
        break :blk op.fetch(silu, .{});
    } else switch (activation) {
        .swiglu => g.sigmoid().mulf(g),
        .geglu_tanh => geluTanh(g),
        else => std.debug.panic("TCL MoE does not support activation {s}", .{@tagName(activation)}),
    };
    var v = activated.mulf(op.fetch(up, .{}));
    if (weights) |w| v = v.mulf(op.fetch(w, .{}));
    return op.commit(v, .{ .dtype = .bf16 });
}

fn geluTanh(g: builder.Value) builder.Value {
    const cube = g.mulf(g).mulf(g);
    const inner = g.addf(cube.mulf(0.044715)).mulf(@sqrt(2.0 / std.math.pi));
    return inner.tanh().addf(1.0).mulf(g).mulf(0.5);
}

/// Furiosa's compiler configuration for MoE kernels (`qwen3_moe/config.py`):
/// Use IO-bound tactics for per-route and FP8 small-batch schedules and
/// compute-bound tactics for blockwise.
/// Keep the compiler's default lowering mode and omit the adaptive-einsum and
/// half-VRF settings removed in TCC 2026.5.
fn configure(b: *tcl.Builder, comptime schedule: Schedule) void {
    const config = comptime blk: {
        const common = .{
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
            .propagate_sparse_axis_from_op = "None",
        };
        const tactics = switch (schedule) {
            .blockwise => .{
                .tactic_hint = .{ .ForLlmModelComputeBound = 200 },
                .dma_preference = 1.2,
            },
            .per_route, .small_batch => .{
                .tactic_hint = .{ .ForLlmModelIOBound = 200 },
                .num_transaction_simulation_per_pe = 1024,
                .dma_preference = 0.8,
            },
        };
        // The transaction-simulation setting is only emitted for IO-bound schedules.
        const common_info = @typeInfo(@TypeOf(common)).@"struct";
        const tactics_info = @typeInfo(@TypeOf(tactics)).@"struct";
        const Config = @Struct(
            .auto,
            null,
            common_info.field_names ++ tactics_info.field_names,
            common_info.field_types ++ tactics_info.field_types,
            common_info.field_attrs ++ tactics_info.field_attrs,
        );
        var merged: Config = undefined;
        for (common_info.field_names) |field| @field(merged, field) = @field(common, field);
        for (tactics_info.field_names) |field| @field(merged, field) = @field(tactics, field);
        break :blk merged;
    };
    b.compilerConfig(config);
}

/// Gather the selected experts for every token.
fn perRoute(b: *tcl.Builder, a: Axes, in: Args, activation: ActivationKind, comptime format: Format) tcl.FinishError!void {
    const gate_up = try b.reshape(in.gate_up, &.{ a.E, a.S2, a.I, a.D }, .{});
    const gu = try b.gather(gate_up, in.ids, a.E, &.{ a.T, a.K, a.S2, a.I, a.D }, .{});
    const dn = try b.gather(in.down, in.ids, a.E, &.{ a.T, a.K, a.D, a.I }, .{});

    const scales = try gatherScales(b, a, in, in.ids, &.{ a.T, a.K }, format);
    const h = try project(b, in.x, gu, scales.gate_up, a.D, a.I, &.{ a.T, a.K, a.S2, a.I }, if (format == .fp8_block128) .bf16 else null, format);
    const act = try gated(b, a, h, &.{ a.T, a.K }, if (format == .fp8_block128) null else in.w, activation, format);
    if (format == .fp8_block128) {
        const y = try project(b, act, dn, scales.down, a.I, a.D, &.{ a.T, a.K, a.D }, .f32, format);
        const op = b.tensorOperation(.{});
        b.ret(&.{try op.commit(op.fetch(y, .{}).mulf(op.fetch(in.w, .{})).reduce(&.{a.K}, .addf), .{ .dtype = .bf16 })});
    } else {
        // The BF16 recipe weights activations and contracts experts together.
        const op = b.tensorOperation(.{});
        b.ret(&.{try op.commit(op.contract(act, dn, &.{ a.T, a.D }), .{ .dtype = .bf16 })});
    }
}

/// Rows per expert block: the next power of two of the routes per expert.
fn blockRows(cfg: Cfg) i64 {
    const per_expert = std.math.divCeil(i64, cfg.num_tokens * cfg.top_k, cfg.local_num_experts) catch unreachable;
    return @intCast(std.math.ceilPowerOfTwo(u64, @intCast(@max(per_expert, 4))) catch unreachable);
}

const blocks_per_chunk = 32;

/// Group routes into expert-major blocks, then run both projections in a
/// device loop over chunks of blocks.
fn blockwise(b: *tcl.Builder, a: Axes, in: Args, cfg: Cfg, comptime format: Format) tcl.FinishError!void {
    const bb = blockRows(cfg);
    // One block per expert at least, then enough for every route.
    const max_blocks = std.math.divCeil(i64, @max(cfg.num_tokens * cfg.top_k - cfg.local_num_experts, 0), bb) catch unreachable;
    const chunks = std.math.divCeil(i64, max_blocks + cfg.local_num_experts, blocks_per_chunk) catch unreachable;
    const BB = b.axis("BB", bb);
    const BGi = b.axis("BGi", blocks_per_chunk);
    const Go = b.axis("Go", chunks);
    const BG = b.axis("BG", chunks * blocks_per_chunk);
    const GB = b.axis("GB", chunks * blocks_per_chunk * bb);
    const T, const K, const E, const D, const I = .{ a.T, a.K, a.E, a.D, a.I };

    // Routing: route (t, k) goes to row `slot` of block `start[e] + rank`.
    const experts = try b.arange(cfg.local_num_experts, .i32, &.{E}, .{});
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
    const block_expert = try op.commit(op.fetch(ended, .{}).reduce(&.{E}, .addi).binary(.mini, cfg.local_num_experts - 1), .{});
    const used = try b.symExpr(.div, try b.symExpr(.add, try b.reduceMaxI32(block_end), blocks_per_chunk - 1), blocks_per_chunk);

    // Rows of unused slots are never read back.
    const rows = try b.reshape(try b.scatter(in.x, slot, GB, &.{ GB, D }, .{}), &.{ Go, BGi, BB, D }, .{});
    const chunk_experts = try b.reshape(block_expert, &.{ Go, BGi }, .{});
    const gate_up = try b.reshape(in.gate_up, &.{ E, a.S2, I, D }, .{});

    var loop = b.openFor(used, .{try b.scratchpad(if (format == .fp8_block128) .f32 else .bf16, &.{ Go, BGi, BB, D })});
    {
        const chunk = try b.indexRead(rows, loop.iv);
        const ids = try b.indexRead(chunk_experts, loop.iv);
        const gu = try b.gather(gate_up, ids, E, &.{ BGi, a.S2, I, D }, .{});
        const dn = try b.gather(in.down, ids, E, &.{ BGi, D, I }, .{});
        const scales = try gatherScales(b, a, in, ids, &.{BGi}, format);
        const h = try project(b, chunk, gu, scales.gate_up, D, I, &.{ BGi, BB, a.S2, I }, if (format == .fp8_block128) .bf16 else null, format);
        const act = try gated(b, a, h, &.{ BGi, BB }, null, cfg.activation, format);
        const y = try project(b, act, dn, scales.down, I, D, &.{ BGi, BB, D }, if (format == .fp8_block128) .f32 else .bf16, format);
        try loop.yield(.{try b.indexWrite(loop.carried[0], loop.iv, y)});
    }
    const out = loop.results[0];

    const per_route = try b.gather(try b.reshape(out, &.{ GB, D }, .{}), slot, GB, &.{ T, K, D }, .{});
    op = b.tensorOperation(.{});
    b.ret(&.{try op.commit(op.fetch(per_route, .{ .typecast_to = .f32 }).mulf(op.fetch(in.w, .{})).reduce(&.{K}, .addf), .{ .dtype = .bf16 })});
}

/// Number of experts processed per device-loop iteration, based on the
/// batch's route count (`cfg.num_tokens * cfg.top_k`).
fn expertsPerChunk(cfg: Cfg) i64 {
    const routes: u64 = @intCast(std.math.divCeil(i64, cfg.num_tokens * cfg.top_k, 16) catch unreachable);
    return @intCast(std.math.clamp(std.math.ceilPowerOfTwo(u64, routes) catch unreachable, 2, 16));
}

/// Process the FP8 experts selected by the batch in groups sized by `expertsPerChunk`.
/// Load each selected expert once and run it for every token. Routing weights
/// scale the F32 down-projection results before summing the experts and steps.
fn fp8SmallBatch(b: *tcl.Builder, a: Axes, in: Args, cfg: Cfg) tcl.FinishError!void {
    const chunk = expertsPerChunk(cfg);
    const steps = std.math.divCeil(i64, @min(cfg.local_num_experts, cfg.num_tokens * cfg.top_k), chunk) catch unreachable;
    const C = b.axis("C", chunk);
    const Go = b.axis("Go", steps);
    const J = b.axis("J", steps * chunk);
    const T, const K, const E, const D, const I = .{ a.T, a.K, a.E, a.D, a.I };

    // Each token's routing weight per expert, and the used experts.
    const experts = try b.arange(cfg.local_num_experts, .i32, &.{E}, .{});
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
        const scales = try gatherScales(b, a, in, ids, &.{C}, .fp8_block128);
        const h = try project(b, in.x, gu, scales.gate_up, D, I, &.{ C, T, a.S2, I }, .bf16, .fp8_block128);
        const act = try gated(b, a, h, &.{ C, T }, null, cfg.activation, .fp8_block128);
        const per_expert = try project(b, act, dn, scales.down, I, D, &.{ C, T, D }, .f32, .fp8_block128);
        op = b.tensorOperation(.{});
        const y = try op.commit(op.fetch(per_expert, .{}).mulf(op.fetch(weights, .{})).reduce(&.{C}, .addf), .{ .dtype = .f32 });
        try loop.yield(.{try b.indexWrite(loop.carried[0], loop.iv, y)});
    }
    op = b.tensorOperation(.{});
    b.ret(&.{try op.commit(op.fetch(loop.results[0], .{}).reduce(&.{Go}, .addf), .{ .dtype = .bf16 })});
}
