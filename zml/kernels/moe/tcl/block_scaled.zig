//! NVFP4 and MXFP8 routed projections. Only a tile of selected expert weights
//! is expanded to BF16 at a time, including in the multi-token prefill graph.
const std = @import("std");

const builder = @import("dsl/tcl/builder");
const stdx = @import("stdx");
const zml = @import("zml/core");
const api = @import("zml/kernels/moe/api");
const nn = @import("zml/nn");
const quant = @import("zml/quantization");

const Tensor = zml.Tensor;
const tcl = zml.kernel.tcl;

pub fn fusedExperts(input: Tensor, ids: Tensor, weights: Tensor, gate_up: nn.Linear, down: nn.Linear, opts: api.Options) Tensor {
    const scheme = gate_up.quantizationScheme().?;
    stdx.debug.assert(down.quantizationScheme() == scheme, "TCL expert projection formats differ", .{});
    const routes = ids.reshape(.{ .route = ids.count() });
    // Stop after the final live route. Padding within a tile reuses an expert
    // already selected on this rank. A completely inactive rank takes zero trips.
    const active = routes.cmp(.GE, .scalar(0, .i32));
    const order = Tensor.iota(routes.shape(), .route);
    const count = order.addConstant(1).mask(active, 0).max(.route);
    const fallback = routes.max(.route).maximum(.scalar(0, .i32)).broad(routes.shape());
    const safe = active.select(routes, fallback);
    const gate = weights.reshape(.{ .route = routes.dim(.route) }).convert(.f32);
    const tokens = activationRows(input, .d, scheme, opts);
    const token = order.divByConst(ids.dim(.topk));
    const first = projection(.{
        .values = tokens.values.gather(.{ .t = token }, .{}),
        .outer = if (tokens.outer) |outer| outer.gather(.{ .t = token }, .{}) else null,
        .amax = if (tokens.amax) |amax| amax.gather(.{ .t = token }, .{}) else null,
    }, safe, count, gate_up, opts.furiosa_scales.gate_up);
    var activated = opts.activation.applyActivation(first, .concatenated);
    if (opts.routing_weight_placement == .before_down)
        activated = activated.mul(gate.broad(activated.shape()));
    const intermediate = activationRows(activated, .out, scheme, opts);
    var result = projection(intermediate, safe, count, down, opts.furiosa_scales.down);
    if (opts.routing_weight_placement == .after_down)
        result = result.convert(.f32).mul(gate.broad(result.shape().withDtype(.f32)));
    const ordered = result.mask(active, 0).convert(.f32);
    return ordered.reshape(.{ .t = ids.dim(.t), .topk = ids.dim(.topk), .d = result.dim(.out) })
        .sum(.topk).squeeze(.topk).convert(input.dtype());
}

const Rows = struct { values: Tensor, outer: ?Tensor = null, amax: ?Tensor = null };

fn activationRows(input: Tensor, axis: anytype, scheme: quant.Quantization.Scheme, opts: api.Options) Rows {
    const rounded = roundBf16(input);
    if (!opts.quantize_input) return .{ .values = rounded };
    const rows = rounded.optimizationBarrier().convert(.f32);
    return switch (scheme) {
        .nvfp4 => nvfp4Rows(rows, axis, opts.nvfp4_input_scale),
        .mxfp8 => .{ .values = mxfp8Rows(rows, axis).convert(.bf16) },
        else => unreachable,
    };
}

/// Preserve the BF16 boundary across fused activation and scale arithmetic.
/// One BF16 ULP can change which NVFP4 candidate wins a whole block.
fn roundBf16(input: Tensor) Tensor {
    if (input.dtype() == .bf16) return input;
    const bits = input.bitCast(.u32);
    const odd = bits.shiftRightLogical(.scalar(16, .u32)).logical(.AND, .scalar(1, .u32));
    return bits.addConstant(0x7fff).add(odd).logical(.AND, .scalar(0xffff0000, .u32))
        .bitCast(.f32).convert(.bf16);
}

/// E2M1 times E4M3 fits exactly in BF16. Keep the non-power-of-two outer
/// scale in FP32 until after contraction, as in the checkpoint's NVFP4 GEMMs.
fn nvfp4Rows(rows: Tensor, axis: anytype, strategy: quant.Nvfp4QuantizationStrategy) Rows {
    const amax = rows.abs().max(axis).squeeze(axis);
    const outer = amax.scale(1.0 / 1536.0);
    const encoder = divideRounded(1536, amax);
    const decoder = divideRounded(1, encoder);
    const blocks = rows.splitAxis(axis, .{ .group = -1, .lane = 16 });
    const maximum = blocks.abs().max(.lane).squeeze(.lane);
    const six = nvfp4Candidate(blocks, maximum, encoder, decoder, outer, 6);
    const values = switch (strategy) {
        .largest_maps_to_six => six.values,
        .best_of_four_or_six => selection_blk: {
            const four = nvfp4Candidate(blocks, maximum, encoder, decoder, outer, 4);
            const better = four.loss.cmp(.LT, six.loss).broad(blocks.shape().withDtype(.bool));
            break :selection_blk better.select(four.values, six.values);
        },
    };
    return .{ .values = values.reshape(rows.shape()).convert(.bf16), .outer = outer, .amax = amax };
}

fn nvfp4Candidate(blocks: Tensor, maximum: Tensor, encoder: Tensor, decoder: Tensor, outer: Tensor, divisor: f32) struct { values: Tensor, loss: Tensor } {
    const raw = maximum.mul(encoder.scale(1.0 / divisor).broad(maximum.shape())).minimum(.scalar(448, .f32));
    const scale = raw.convert(.f8e4m3fn).optimizationBarrier().convert(.f32);
    const denominator = scale.mul(decoder.broad(scale.shape()));
    const inverse = divideRounded(1, denominator);
    const values = quant.roundE2m1(blocks.mul(inverse.broad(blocks.shape()))).mul(scale.broad(blocks.shape()));
    var loss = values.mul(outer.broad(values.shape())).sub(blocks).abs();
    // Match the adjacent-pair reduction used to break close four-versus-six decisions.
    inline for (.{ 8, 4, 2, 1 }) |pairs|
        loss = loss.splitAxis(.lane, .{ .pairs = pairs, .pair = 2 }).sum(.pair).squeeze(.pair).rename(.{ .pairs = .lane });
    return .{ .values = values, .loss = loss.squeeze(.lane) };
}

/// RNGD's FP32 divider can miss by an ULP (1 / 0.46875, for example).
/// NVFP4 first rounds this division, then multiplies by the input: a single
/// ULP in the reciprocal can move a BF16 value across an E2M1 midpoint.
fn divideRounded(comptime numerator: f32, denominator: Tensor) Tensor {
    comptime std.debug.assert(numerator == 1 or numerator == 1536);
    const estimate = Tensor.scalar(numerator, .f32).div(denominator).minimum(.scalar(std.math.floatMax(f32), .f32));
    const numerator_bits: u32 = @bitCast(numerator);
    const bits = denominator.bitCast(.i32);
    const exponent = bits.shiftRightLogical(.scalar(23, .i32));
    const mantissa = bits.logical(.AND, .scalar(0x7fffff, .i32)).addConstant(0x800000);
    const extra = mantissa.cmp(.GT, .scalar(0x800000 | (numerator_bits & 0x7fffff), .i32)).convert(.i32);
    const result_exponent = Tensor.scalar(127 + (numerator_bits >> 23), .i32).sub(exponent).sub(extra);
    const base = result_exponent.shiftLeft(.scalar(23, .i32));
    var candidate = estimate.bitCast(.i32).sub(base).addConstant(0x800000)
        .clamp(.scalar(0x800000, .i32), .scalar(0xffffff, .i32));
    // For these numerators the normalized integer dividend's low 32 bits
    // are zero. The nearby quotient therefore gives the exact signed
    // remainder in one wrapping multiplication, without a 64-bit division.
    inline for (0..2) |_| {
        const twice = candidate.mul(mantissa).mul(.scalar(2, .i32));
        const odd = candidate.logical(.AND, .scalar(1, .i32));
        const up = twice.add(mantissa).cmp(.LT, odd).convert(.i32);
        const down = twice.add(odd).cmp(.GT, mantissa).convert(.i32);
        candidate = candidate.add(up).sub(down);
    }
    const normal = exponent.cmp(.GT, .scalar(0, .i32)).logical(.AND, result_exponent.cmp(.GT, .scalar(0, .i32)))
        .logical(.AND, result_exponent.cmp(.LT, .scalar(255, .i32)));
    return normal.select(base.add(candidate.subConstant(0x800000)).bitCast(.f32), estimate).optimizationBarrier();
}

fn mxfp8Rows(rows: Tensor, axis: anytype) Tensor {
    const blocks = rows.splitAxis(axis, .{ .group = -1, .lane = 32 });
    const raw = blocks.abs().max(.lane).scale(1.0 / 448.0);
    const minimum = Tensor.scalar(0x1p-127, .f32);
    const scale = raw.cmp(.LE, minimum).select(minimum, quant.ceilPowerOfTwo(raw).minimum(.scalar(0x1p127, .f32)));
    const inverse = raw.cmp(.LE, minimum).select(.scalar(0x1p127, .f32), Tensor.scalar(1, .f32).div(scale));
    const values = blocks.mul(inverse.broad(blocks.shape())).clamp(.scalar(-448, .f32), .scalar(448, .f32))
        .convert(.f8e4m3fn).optimizationBarrier().convert(.f32);
    return values.mul(scale.broad(blocks.shape())).reshape(rows.shape());
}

const Format = enum { nvfp4, mxfp8 };
const Config = struct {
    format: Format,
    experts: i64,
    routes: i64,
    lanes: i64,
    width: i64,
    outputs: i64,
};

fn projection(
    rows: Rows,
    ids: Tensor,
    count: Tensor,
    linear: nn.Linear,
    factors_: ?api.Options.FuriosaScale,
) Tensor {
    const input = rows.values;
    const metadata = linear.quantization.?;
    const format: Format = if (metadata.scheme == .nvfp4) .nvfp4 else .mxfp8;
    const width = input.dim(1);
    const outputs = linear.weight.dim(1);
    const experts = linear.weight.dim(0);
    const block: i64 = if (format == .nvfp4) 16 else 32;
    const weight = if (linear.weight.dtype() == .f4e2m1)
        linear.weight.reshape(.{ experts, outputs, @divExact(width, 2), 2 }).bitCast(.u8)
    else
        linear.weight;
    stdx.debug.assert(weight.dtype() == (if (format == .nvfp4) @as(zml.DataType, .u8) else .f8e4m3fn) and
        weight.dim(2) == (if (format == .nvfp4) @divExact(width, 2) else width) and
        !metadata.swizzled_scales and metadata.scales.dim(0) == experts and
        metadata.scales.dim(1) == outputs and metadata.scales.dim(2) * block == width, "TCL expects plain per-block expert weights and scales, got {f} and {f}", .{ weight.shape(), metadata.scales.shape() });
    const scales = if (format == .mxfp8) alignedScaleBytes(metadata.scales) else metadata.scales;
    const shape = zml.Shape.init(.{ .route = ids.count(), .out = outputs }, .f32);
    const cfg: Config = .{
        .format = format,
        .experts = experts,
        .width = width,
        .outputs = outputs,
        .routes = ids.dim(.route),
        .lanes = @intCast(std.math.gcd(@as(u64, @intCast(ids.count())), 4)),
    };
    var output = Project.call(.{ .rows = input, .ids = ids, .count = count, .weight = weight, .scales = scales }, .{ .output = shape }, .{ .cfg = cfg }).output;
    if (factors_) |factors| {
        if (rows.amax) |amax| {
            const weight_amax = factors.amax.gather(.{ .expert = ids }, .{}).convert(.f32);
            const divisor = factors.denominator.asScalar().convert(.f32);
            const product = amax.mul(weight_amax).optimizationBarrier();
            const scale = product.mul(divideRounded(1, divisor.scale(1536)));
            return roundBf16(output.mul(scale.broad(output.shape())));
        }
    }
    if (metadata.global_scale) |outer| {
        const scale = outer.asMultiplier();
        if (scale.rank() == 0) {
            output = output.mul(scale);
        } else {
            const selected = scale.gather(.{ .expert = ids }, .{});
            if (scale.rank() == 2) {
                output = output.splitAxis(.out, .{ .proj = 2, .half = @divExact(outputs, 2) });
            }
            output = output.mul(selected.broad(output.shape())).reshape(shape);
        }
    }
    if (rows.outer) |scale| output = output.mul(scale.broad(output.shape()));
    return roundBf16(output);
}

const Project = tcl.Kernel(Config, .{
    .name = "moe_block_scaled_projection",
    .inputs = &.{ "rows", "ids", "count", "weight", "scales" },
    .outputs = &.{"output"},
    .run = emit,
});

/// E8M0 rows are read in groups of four bytes after selecting the expert.
fn alignedScaleBytes(scales: Tensor) Tensor {
    const bytes = scales.bitCast(.u8);
    const width = bytes.dim(2);
    const padding = @mod(-width, 4);
    if (padding == 0) return bytes;
    return bytes.withTags(.{ .expert, .out, .byte }).pad(0, .{ .byte = Tensor.Pad{ .high = padding } });
}

fn emit(b: *tcl.Builder, cfg: Config) tcl.FinishError!void {
    const expert = b.axis("Expert", cfg.experts);
    const route = b.axis("Route", cfg.routes);
    const tile = b.axis("Tile", @divExact(cfg.routes, cfg.lanes));
    const lane = b.axis("Lane", cfg.lanes);
    const width = b.axis("Width", cfg.width);
    const output = b.axis("Output", cfg.outputs);
    const nvfp4 = cfg.format == .nvfp4;
    const stored = b.axis("Stored", if (nvfp4) @divExact(cfg.width, 2) else cfg.width);
    const block = b.axis("Block", @as(i64, if (nvfp4) 16 else 32));
    const group = b.axis("Group", @divExact(cfg.width, block.size));
    const scale_columns = if (nvfp4) group else b.axis("ScaleByte", @divTrunc(group.size + 3, 4) * 4);
    const inputs = try b.declareArgs(.{
        .rows = .{ .dtype = .bf16, .axes = &.{ route, width } },
        .ids = .{ .dtype = .i32, .axes = &.{route} },
        .count = .{ .dtype = .i32, .axes = &.{b.axis("One", 1)} },
        .weight = .{ .dtype = if (nvfp4) tcl.DType.u8 else .f8_e4, .axes = &.{ expert, output, stored } },
        .scales = .{ .dtype = if (nvfp4) tcl.DType.f8_e4 else .u8, .axes = &.{ expert, output, scale_columns } },
    });
    const rows = try b.reshape(inputs.rows, &.{ tile, lane, width }, .{});
    const ids = try b.reshape(inputs.ids, &.{ tile, lane }, .{});
    // Gather E8M0 in word units. TCC cannot align a byte gather followed by
    // a transmute when there are only four scale bytes per output channel.
    const scale_axis = if (nvfp4) group else b.axis("ScaleWord", @divExact(scale_columns.size, 4));
    const scale_table = if (nvfp4) inputs.scales else try b.transmute(inputs.scales, .i32, &.{ expert, output, scale_axis }, .{});
    const trips = try b.symExpr(.div, try b.symExpr(.add, try b.reduceMaxI32(inputs.count), cfg.lanes - 1), cfg.lanes);
    var loop = b.openFor(trips, .{try b.full(0, .f32, &.{ tile, lane, output })});
    {
        const selected = try b.indexRead(ids, loop.iv);
        const payload = try b.gather(inputs.weight, selected, expert, &.{ lane, output, stored }, .{});
        const raw_scale = try b.gather(scale_table, selected, expert, &.{ lane, output, scale_axis }, .{});
        const decoded = if (nvfp4) decoded_blk: {
            const word = b.axis("WeightWord", @divExact(stored.size, 4));
            const words = try b.transmute(payload, .i32, &.{ lane, output, word }, .{});
            break :decoded_blk try e2m1(b, try unpackCodes(b, words, 4));
        } else payload;
        const values = try b.reshape(decoded, &.{ lane, output, group, block }, .{});
        const scale = if (nvfp4) raw_scale else scale_blk: {
            const padded = scale_columns;
            const codes = try b.reshape(try unpackCodes(b, raw_scale, 8), &.{ lane, output, padded }, .{});
            const axes = &.{ lane, output, group };
            const exact = if (padded.size == group.size) try b.reshape(codes, axes, .{}) else try b.slice(codes, padded, 0, axes, .{});
            break :scale_blk try exponentScale(b, exact);
        };
        var op = b.tensorOperation(.{ .tactic = .elementwise });
        const scaled = try op.commit(op.fetch(values, .{}).mulf(op.fetch(scale, .{})), .{ .dtype = .bf16 });
        const weights = try b.reshape(scaled, &.{ lane, output, width }, .{});
        const activations = try b.indexRead(rows, loop.iv);
        op = b.tensorOperation(.{});
        const products = try op.commit(op.contract(activations, weights, &.{ lane, output }), .{ .dtype = .f32 });
        try loop.yield(.{try b.indexWrite(loop.carried[0], loop.iv, products)});
    }
    b.ret(&.{try b.reshape(loop.results[0], &.{ route, output }, .{})});
}

/// E8M0 stores the biased binary exponent, including the smallest power at code 0.
fn exponentScale(b: *tcl.Builder, raw: builder.Tensor) tcl.FinishError!builder.Tensor {
    var op = b.tensorOperation(.{});
    const exponent = try op.commit(op.fetch(raw, .{}).binary(.shl, 23), .{});
    op = b.tensorOperation(.{});
    const zero = try op.commit(op.fetch(raw, .{}).subi(1).binary(.shr_logical, 31), .{});
    op = b.tensorOperation(.{});
    // Both exceptional exponent bytes set mantissa bit 22: 2^-127 or quiet NaN.
    return op.commit(op.fetch(raw, .{}).addi(1).binary(.shr_logical, 8)
        .addi(op.fetch(zero, .{})).binary(.shl, 22).addi(op.fetch(exponent, .{}))
        .unary(.reinterpret_as_float), .{});
}

/// TCC's VE cannot fetch a U8 as I32. Read packed words and extract fields
/// in little-endian checkpoint order on the selected route tile.
fn unpackCodes(b: *tcl.Builder, packed_words: builder.Tensor, comptime width: i64) tcl.FinishError!builder.Tensor {
    const field = b.axis("Field", @divExact(32, width));
    const shifts = try b.arange(32, .i32, &.{field}, .{ .step = width });
    const axes = &.{ packed_words.axes[0], packed_words.axes[1], packed_words.axes[2], field };
    var op = b.tensorOperation(.{});
    const expanded = try op.commit(op.fetch(packed_words, .{ .broadcast_to = axes }), .{});
    op = b.tensorOperation(.{ .tactic = .elementwise });
    return op.commit(op.fetch(expanded, .{})
        .binary(.shr_logical, op.fetch(shifts, .{})).binary(.bit_andi, (1 << width) - 1), .{});
}

/// E2M1 magnitudes in units of 0.5 are 0, 1, 2, 3, 4, 6, 8, 12.
fn e2m1(b: *tcl.Builder, nibble: builder.Tensor) tcl.FinishError!builder.Tensor {
    var op = b.tensorOperation(.{});
    const magnitude = try op.commit(op.fetch(nibble, .{}).binary(.bit_andi, 7), .{});
    op = b.tensorOperation(.{});
    const middle = try op.commit(op.fetch(magnitude, .{}).subi(4).binary(.maxi, 0).addi(op.fetch(magnitude, .{})), .{});
    op = b.tensorOperation(.{});
    const absolute = try op.commit(op.fetch(magnitude, .{}).subi(6).binary(.maxi, 0).muli(2)
        .addi(op.fetch(middle, .{})).unary(.to_float).mulf(0.5), .{});
    op = b.tensorOperation(.{});
    const negative = try op.commit(op.fetch(absolute, .{}).unary(.negf), .{});
    op = b.tensorOperation(.{});
    return op.commit(op.where(nibble, .lt, 8, op.fetch(absolute, .{}), op.fetch(negative, .{})), .{ .dtype = .bf16 });
}
