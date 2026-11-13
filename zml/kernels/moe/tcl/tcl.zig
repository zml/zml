//! Furiosa RNGD fused experts in TCL, after Furiosa's production kernels.
const std = @import("std");

const stdx = @import("stdx");

const zml = @import("../../../zml.zig");
const Tensor = zml.Tensor;
const tcl = zml.kernel.tcl;
const kernels = @import("moe.zig");
const api = @import("../api.zig");

/// `input` is `{.t, .d}`, routing IDs and weights are `{.t, .topk}`, `gate_up`
/// is `{.expert, .dout = 2 * inter (gate then up), .d}`, and `down` is
/// `{.expert, .d, .dout = inter}`. Experts are BF16 or block-128 FP8 with
/// BF16/F32 scales. Supports plain SwiGLU and BF16 GEGLU with tanh GELU.
pub fn fusedExperts(input: Tensor, routing: api.LocalRouting, gate_up_linear: zml.nn.Linear, down_linear: zml.nn.Linear, opts: api.Options) Tensor {
    const topk_ids = routing.localIds();
    stdx.debug.assert(!opts.quantize_input and opts.routing_weight_placement == .after_down, "TCL MoE requires quantize_input=false and routing_weight_placement=after_down", .{});
    stdx.debug.assert(gate_up_linear.bias == null and down_linear.bias == null, "TCL MoE takes experts without bias", .{});
    stdx.debug.assert(input.dtype() == .bf16 and topk_ids.dtype() == .i32, "TCL MoE takes BF16 activations and i32 ids", .{});
    const format = expertFormat(gate_up_linear, down_linear);
    const swiglu = opts.activation == .swiglu and std.meta.eql(opts.activation.swiglu, .{});
    stdx.debug.assert(swiglu or (format == .bf16 and opts.activation == .geglu_tanh), "TCL MoE supports plain SwiGLU or GEGLU-tanh with BF16 experts, got {s}", .{@tagName(opts.activation)});

    const gate_up = gate_up_linear.weight;
    const down = down_linear.weight;
    const cfg: kernels.Cfg = .{
        .num_tokens = input.dim(.t),
        .top_k = topk_ids.dim(.topk),
        .local_num_experts = gate_up.dim(.expert),
        .hidden_size = input.dim(.d),
        .expert_intermediate_size = down.dim(.dout),
        .weight_scale_dtype = switch (format) {
            .bf16 => .bf16,
            .fp8_block128 => |s| tcl.from(s.gate_up.dtype()),
        },
        .activation = std.meta.activeTag(opts.activation),
    };
    stdx.debug.assert(gate_up.dim(.dout) == 2 * cfg.expert_intermediate_size and down.dim(.d) == cfg.hidden_size and gate_up.dim(.d) == cfg.hidden_size, "TCL MoE: mismatched expert shapes", .{});

    const weights = routing.weights.convert(.f32);
    const local_weights = switch (format) {
        .bf16 => weights,
        .fp8_block128 => topk_ids.cmp(.GE, .scalar(0, .i32)).select(weights, .scalar(0, .f32)),
    };
    return switch (format) {
        .bf16 => selectScheduleAndRun(.bf16, cfg, .{ .x = input, .ids = topk_ids, .w = local_weights, .gate_up = gate_up, .down = down }),
        .fp8_block128 => |s| selectScheduleAndRun(.fp8_block128, cfg, .{
            .x = input,
            .ids = topk_ids,
            .w = local_weights,
            .gate_up = gate_up,
            .down = down,
            .gate_up_scale = s.gate_up,
            .down_scale = s.down,
        }),
    };
}

const Scales = struct { gate_up: Tensor, down: Tensor };

const Format = union(kernels.Format) {
    bf16: void,
    fp8_block128: Scales,
};

fn expertFormat(gate_up: zml.nn.Linear, down: zml.nn.Linear) Format {
    const scheme = gate_up.quantizationScheme();
    stdx.debug.assert(scheme == down.quantizationScheme(), "TCL MoE projections must have matching quantization schemes", .{});
    return switch (scheme orelse {
        stdx.debug.assert(gate_up.weight.dtype() == .bf16 and down.weight.dtype() == .bf16, "TCL MoE unquantized expert weights must be BF16", .{});
        return .bf16;
    }) {
        .fp8_block128 => blk: {
            const scales: Scales = .{ .gate_up = fp8Scales(gate_up), .down = fp8Scales(down) };
            stdx.debug.assert(scales.gate_up.dtype() == scales.down.dtype(), "TCL MoE projection scales must have matching types", .{});
            break :blk .{ .fp8_block128 = scales };
        },
        else => |unsupported| stdx.debug.panic("TCL MoE does not support quantization scheme {s}", .{@tagName(unsupported)}),
    };
}

fn fp8Scales(linear: zml.nn.Linear) Tensor {
    const q = linear.quantization.?;
    stdx.debug.assert(linear.weight.dtype() == .f8e4m3fn and !q.swizzled_scales and q.global_scale == null and q.input_scale == null, "TCL MoE takes plain E4M3FN block-128 weights", .{});
    stdx.debug.assert(q.scales.dtype() == .bf16 or q.scales.dtype() == .f32, "TCL MoE FP8 scales must be BF16 or F32", .{});
    const w = linear.weight;
    const scale = q.scales;
    stdx.debug.assert(@mod(w.dim(.d), 128) == 0 and @mod(w.dim(.dout), 128) == 0 and scale.dim(.expert) == w.dim(.expert) and scale.dim(.d) * 128 == w.dim(.d) and scale.dim(.dout) * 128 == w.dim(.dout), "TCL MoE FP8 scales must cover 128x128 weight blocks", .{});
    return scale;
}

// Maximum total input tokens per MoE call before padding for the FP8
// small-batch schedule. It loads each used expert's weights once and reuses
// them across the batch.
// Larger batches fall back to per-route or blockwise dispatch.
const fp8_small_batch_max_tokens = 64;

fn selectScheduleAndRun(comptime format: kernels.Format, cfg: kernels.Cfg, inputs: kernels.Kernel(format, .per_route).Inputs) Tensor {
    if (format == .fp8_block128 and cfg.num_tokens <= fp8_small_batch_max_tokens) {
        return run(format, .small_batch, cfg, inputs);
    }
    if (cfg.num_tokens * cfg.top_k <= cfg.local_num_experts) {
        return run(format, .per_route, cfg, inputs);
    }
    return run(format, .blockwise, cfg, inputs);
}

fn run(comptime format: kernels.Format, comptime schedule: kernels.Schedule, cfg: kernels.Cfg, inputs: kernels.Kernel(format, .per_route).Inputs) Tensor {
    const call_cfg, const prepared = if (schedule == .small_batch)
        prepareFp8SmallBatch(cfg, inputs)
    else
        .{ cfg, prepareRoutedInputs(cfg, inputs) };

    const shape = prepared.x.shape();
    var y = kernels.Kernel(format, schedule).call(prepared, .{ .y = shape }, .{ .cfg = call_cfg }).y;

    if (schedule == .small_batch) y = y.slice(0, .{ .end = cfg.num_tokens });
    return y.reshape(inputs.x.shape());
}

fn prepareRoutedInputs(cfg: kernels.Cfg, inputs: anytype) @TypeOf(inputs) {
    var prepared = inputs;
    prepared.x = inputs.x.reshape(.{ cfg.num_tokens, cfg.hidden_size });
    const ids = inputs.ids.reshape(.{ cfg.num_tokens, cfg.top_k });
    const w = inputs.w.reshape(.{ cfg.num_tokens, cfg.top_k });
    // These kernels gather every route, and blockwise also scatters each route
    // into an expert slot. Give nonlocal routes a valid slot and zero weight.
    const valid = ids.cmp(.GE, Tensor.scalar(0, .i32)).logical(.AND, ids.cmp(.LT, Tensor.scalar(cfg.local_num_experts, .i32)));
    prepared.ids = valid.select(ids, Tensor.scalar(0, .i32));
    prepared.w = valid.select(w, Tensor.scalar(0, .f32));
    return prepared;
}

fn prepareFp8SmallBatch(cfg: kernels.Cfg, inputs: kernels.Kernel(.fp8_block128, .small_batch).Inputs) struct { kernels.Cfg, @TypeOf(inputs) } {
    // WORKAROUND (TCC 2026.5.0b0): a token count other than a power of two
    // panics its adder tree ("Fail to cover valid range", 5 tokens padded
    // to 8): pad with unrouted tokens.
    var padded = cfg;
    padded.num_tokens = @intCast(std.math.ceilPowerOfTwo(u64, @intCast(cfg.num_tokens)) catch unreachable);
    const pad = padded.num_tokens - cfg.num_tokens;
    var prepared = inputs;
    prepared.x = inputs.x.pad(0, .{ .t = Tensor.Pad{ .high = pad } }).reshape(.{ padded.num_tokens, cfg.hidden_size });
    prepared.ids = inputs.ids.pad(-1, .{ .t = Tensor.Pad{ .high = pad } }).reshape(.{ padded.num_tokens, cfg.top_k });
    prepared.w = inputs.w.pad(0, .{ .t = Tensor.Pad{ .high = pad } }).reshape(.{ padded.num_tokens, cfg.top_k });
    return .{ padded, prepared };
}

test "tcl fused experts run on furiosa" {
    try testFusedExperts(.{ .swiglu = .{} });
    try testFusedExperts(.geglu_tanh);
}

fn testFusedExperts(activation: zml.moe.Activation) !void {
    const platform = zml.testing.env();
    if (platform.target != .furiosa) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const bf16 = zml.floats.BFloat16;

    // With two routes per token and eight experts, four tokens use per-route
    // dispatch and sixteen use blockwise dispatch.
    for ([_]i64{ 4, 16 }) |t| {
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
            pub fn forward(x_: Tensor, ids_: Tensor, w_: Tensor, gu_: Tensor, dn_: Tensor, act: zml.moe.Activation) Tensor {
                return zml.moe.forwardMoe(x_, ids_, w_, .init(gu_, null, .dout), .init(dn_, null, .dout), .tcl, zml.Compiler.current().platform.replicated_mesh, .expert(.replicated), .{
                    .activation = act,
                    .quantize_input = false,
                    .routing_weight_placement = .after_down,
                });
            }
        };
        var exe = try platform.compileFn(allocator, io, Mod.forward, .{ x, ids, w, gu, dn, activation }, .{});
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
                    const gate = switch (activation) {
                        .swiglu => g / (1 + @exp(-g)),
                        .geglu_tanh => 0.5 * g * (1 + std.math.tanh(@sqrt(2.0 / std.math.pi) * (g + 0.044715 * g * g * g))),
                        else => unreachable,
                    };
                    act[n] = gate * u;
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
