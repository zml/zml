const std = @import("std");
const zml = @import("../../zml.zig");
const shared = @import("../fused_experts.zig");
const mxfp4 = @import("mxfp4.zig");
const Tensor = zml.Tensor;
const Shape = zml.Shape;
const fly = zml.kernel.fly;
const Builder = fly.Builder;

pub const Parameters = shared.Parameters;

/// The Fly compiler and its validated MFMA atom on this device. The operand
/// contract is checked separately by supports().
pub fn isAvailable(platform: *const zml.Platform) bool {
    return fly.supportsMmaAtom(platform, mxfp4.matrixInstruction);
}

pub fn supports(opts: shared.FusedExpertsArgs) bool {
    if (opts.hidden_states.dtype() != .bf16 or opts.gate_up.quantizationScheme() != .mxfp4 or
        opts.down.quantizationScheme() != .mxfp4 or opts.gate_up.bias != null or opts.down.bias != null or
        opts.routing_weight_placement != .before_down) return false;
    const gate_up = opts.gate_up.weight;
    const down = opts.down.weight;
    if (gate_up.rank() != 3 or down.rank() != 3 or opts.hidden_states.dim(.d) != 5120) return false;
    const gate_up_scale = opts.gate_up.quantizationScales() orelse return false;
    const down_scale = opts.down.quantizationScales() orelse return false;
    const experts = if (opts.expert_map) |map| map.dim(.expert) else gate_up.dim(0);
    if (experts < 1 or experts > 512) return false;
    inline for (.{ .{ gate_up, gate_up_scale, 4608, 5120 }, .{ down, down_scale, 5120, 2304 } }) |projection| {
        const weight, const scale, const n, const k = projection;
        const packing_factor: i64 = switch (weight.dtype()) {
            .u8, .i8 => 2,
            .f4e2m1 => 1,
            else => return false,
        };
        if (weight.dim(0) < 1 or weight.dim(0) > 512 or weight.dim(1) != n or weight.dim(2) * packing_factor != k or
            scale.rank() != 3 or scale.dim(0) != weight.dim(0) or scale.dim(1) != n or scale.dim(2) != k / 32) return false;
        switch (scale.dtype()) {
            .u8, .i8, .f8e8m0 => {},
            else => return false,
        }
    }
    const routes = opts.hidden_states.dim(.b) * opts.hidden_states.dim(.s) * opts.topk_ids.dim(.top_expert);
    return routes > 0 and routes <= std.math.maxInt(i32) / 5120;
}

pub fn prepareInput(x: Tensor, scheme: ?zml.Quantization.Scheme, _: zml.DataType, _: bool) struct { Tensor, ?Tensor } {
    std.debug.assert(scheme == .mxfp4);
    return .{ x.convert(.bf16), null };
}

pub fn call(opts: shared.GemmOptions) zml.Tensor {
    zml.stdx.debug.assert(opts.quant_scheme == .mxfp4 and opts.bias == null and opts.routing_weights == null and opts.input_scale == null, "Fly MoE requires MXFP4 weights, BF16 activations and routing weights before the down projection", .{});
    const weight = if (opts.weight.dtype() == .f4e2m1)
        opts.weight.reshape(opts.weight.shape().setDim(2, @divExact(opts.weight.dim(2), 2)).append(.{ .nibble = 2 })).bitCast(.u8)
    else
        opts.weight;
    return mxfp4.call(opts.input, weight, opts.weight_scale.?.bitCast(.u8), opts.routing.sorted_token_ids, opts.expert_ids, opts.routing.num_tokens_post_padded, opts.output_shape, opts.top_k, opts.routing.naive_block_assignment, opts.launch_config.block_size_m);
}

pub fn prepareRouting(topk_ids: Tensor, num_experts: i64, block_size_m: i64) shared.Routing {
    std.debug.assert(num_experts > 0 and num_experts <= 512);
    std.debug.assert(block_size_m > 0 and block_size_m <= 128 and @mod(block_size_m, 16) == 0);
    const ids = topk_ids.withTags(.{ .token, .topk }).convert(.i32);
    const num_assignments = ids.dim(.token) * ids.dim(.topk);
    // GEMV handles up to six routes; MFMA shares weights across sorted rows.
    const naive_block_assignment = num_assignments <= 6;
    const max_num_tokens_padded = if (naive_block_assignment or num_assignments < num_experts)
        num_assignments * block_size_m
    else
        (std.math.divCeil(i64, num_assignments + num_experts * (block_size_m - 1), block_size_m) catch unreachable) * block_size_m;
    const sorted_token_ids, const expert_ids, const num_tokens_post_padded = if (naive_block_assignment) blk: {
        break :blk .{
            Tensor.zeroes(Shape.init(.{ .g = 1 }, .i32)),
            ids.reshape(.{ .g = num_assignments }),
            Tensor.constant(.{ .i32 = @as(i32, @intCast(max_num_tokens_padded)) }).reshape(.{1}),
        };
    } else blk: {
        const flat_experts = ids.reshape(.{ .g = num_assignments });
        const aligned = Align.call(
            flat_experts,
            Tensor.zeroes(Shape.init(.{ .g = max_num_tokens_padded }, .i32)),
            Tensor.zeroes(Shape.init(.{ .g = @divExact(max_num_tokens_padded, block_size_m) }, .i32)),
            Tensor.zeroes(Shape.init(.{ .g = 1 }, .i32)),
            Tensor.zeroes(Shape.init(.{ .g = num_experts + 1 }, .i32)),
            block_size_m,
        );
        const sorted = Sort.call(flat_experts, aligned.sorted_token_ids, aligned.cumsum);
        break :blk .{ sorted.sorted_token_ids, aligned.expert_ids, aligned.num_tokens_post_pad };
    };
    return .{
        .sorted_token_ids = sorted_token_ids,
        .expert_ids = expert_ids,
        .num_tokens_post_padded = num_tokens_post_padded,
        .max_num_tokens_padded = max_num_tokens_padded,
        .num_assignments = num_assignments,
        .naive_block_assignment = naive_block_assignment,
    };
}

const Align = struct {
    const Result = struct {
        sorted_token_ids: Tensor,
        expert_ids: Tensor,
        num_tokens_post_pad: Tensor,
        cumsum: Tensor,
    };

    fn supports(ids: Tensor, sorted: Tensor, experts: Tensor, total: Tensor, cumsum: Tensor, block_size: i64) bool {
        for ([_]Tensor{ ids, sorted, experts, total, cumsum }) |tensor| {
            if (tensor.rank() != 1 or tensor.dtype() != .i32) return false;
        }
        const num_experts = cumsum.dim(0) - 1;
        if (num_experts < 1 or num_experts > 512 or total.dim(0) != 1 or ids.dim(0) > std.math.maxInt(i32) or
            block_size <= 0 or block_size > 128 or @mod(block_size, 16) != 0) return false;
        const capacity_bound = if (ids.dim(0) < num_experts)
            ids.dim(0) * block_size
        else
            @divTrunc(ids.dim(0) + num_experts * (block_size - 1) + block_size - 1, block_size) * block_size;
        return capacity_bound <= std.math.maxInt(i32) and sorted.dim(0) >= capacity_bound and
            sorted.dim(0) <= std.math.maxInt(i32) and @mod(sorted.dim(0), block_size) == 0 and
            experts.dim(0) == @divTrunc(sorted.dim(0), block_size);
    }

    /// All four outputs alias their corresponding input buffers. Invalid expert IDs
    /// are ignored, and all unused expert blocks are marked -1. Prefix ends in LDS
    /// let each thread find its output block's expert without scanning every expert.
    fn call(ids: Tensor, sorted: Tensor, experts: Tensor, total: Tensor, cumsum: Tensor, block_size: i64) Result {
        std.debug.assert(Align.supports(ids, sorted, experts, total, cumsum, block_size));
        const cfg: Config = .{
            .routes = ids.dim(0),
            .experts = cumsum.dim(0) - 1,
            .capacity = sorted.dim(0),
            .blocks = experts.dim(0),
            .block_size = block_size,
            .padding = block_size - 1,
        };
        const result = Kernel.call(.{ .ids = ids, .sorted = sorted, .experts = experts, .total = total, .cumsum_in = cumsum }, .{ .sorted_token_ids = sorted.shape(), .expert_ids = experts.shape(), .num_tokens_post_pad = total.shape(), .cumsum = cumsum.shape() }, .{
            .cfg = cfg,
            .threads = 512,
            .grid = .{ 2, 1, 1 },
            .output_operand_aliases = .{ .sorted_token_ids = .sorted, .expert_ids = .experts, .num_tokens_post_pad = .total, .cumsum = .cumsum_in },
        });
        return .{ .sorted_token_ids = result.sorted_token_ids, .expert_ids = result.expert_ids, .num_tokens_post_pad = result.num_tokens_post_pad, .cumsum = result.cumsum };
    }

    const Config = struct {
        routes: i64,
        experts: i64,
        capacity: i64,
        blocks: i64,
        block_size: i64,
        padding: i64,
    };

    const Kernel = zml.kernel.fly.Kernel(Config, .{
        .name = "fly_moe_align",
        .inputs = &.{ "ids", "sorted", "experts", "total", "cumsum_in" },
        .outputs = &.{ "sorted_token_ids", "expert_ids", "num_tokens_post_pad", "cumsum" },
        .run = run,
    });

    fn run(b: *Builder, cfg: Config) fly.FinishError!void {
        const args = Kernel.args(b);
        const tid = b.threadId(.x);
        const tid64 = tid.to(.i64);
        const zero = b.constant(.i32, 0);
        const one = b.constant(.i32, 1);
        const experts = b.constant(.i32, cfg.experts);
        // Distinct arrays match the histogram, inclusive ends and wave totals.
        const counts = b.sharedArray(.i32, 512, 16);
        const ends = b.sharedArray(.i32, 512, 16);
        const waves = b.sharedArray(.i32, 8, 16);
        var fill = b.openIfElse(b.blockId(.x).cmp(.ne, 0), .{});
        var positions = b.openFor(tid64, cfg.capacity, 512, .{});
        args.sorted_token_ids.set(positions.iv, b.constant(.i32, cfg.routes));
        positions.yield(.{});
        fill.yieldThen(.{});

        counts.set(tid, zero);
        b.barrier();
        var routes = b.openFor(tid64, cfg.routes, 512, .{});
        const id = args.ids.at(routes.iv);
        var valid_id = b.openIf(id.cmp(.ge, 0).bitAnd(id.cmp(.lt, cfg.experts)));
        _ = b.ptrAtomicAdd(counts.emitIter().addOffset(id), one, .workgroup);
        valid_id.yieldThen(.{});
        routes.yield(.{});
        b.barrier();

        const padded = counts.at(tid).add(cfg.padding).divUnsigned(cfg.block_size).mul(cfg.block_size);
        const lane = tid.remUnsigned(64);
        const wave = tid.divUnsigned(64);
        var prefix = padded;
        inline for (.{ 1, 2, 4, 8, 16, 32 }) |shift| {
            const peer = prefix.shuffle(.up, shift, 64);
            prefix = prefix.add(lane.cmpUnsigned(.ge, shift).select(peer, zero));
        }
        var last_lane = b.openIf(lane.cmp(.eq, 63));
        waves.set(wave, prefix);
        last_lane.yieldThen(.{});
        b.barrier();

        var sums = b.openFor(zero, 8, 1, .{ zero, zero });
        const value = waves.at(sums.iv);
        const preceding = sums.carried[0].add(sums.iv.cmpUnsigned(.lt, wave).select(value, zero));
        const total = sums.carried[1].add(value);
        sums.yield(.{ preceding, total });
        const end = sums.results[0].add(prefix);
        const start = end.sub(padded);
        ends.set(tid, end);
        var has_expert = b.openIf(tid.cmpUnsigned(.lt, cfg.experts));
        args.cumsum.set(tid64, start);
        has_expert.yieldThen(.{});
        var first = b.openIf(tid.cmp(.eq, 0));
        args.cumsum.set(b.constant(.i64, cfg.experts), sums.results[1]);
        args.num_tokens_post_pad.set(b.constant(.i64, 0), sums.results[1]);
        first.yieldThen(.{});
        b.barrier();

        var blocks = b.openFor(tid64, cfg.blocks, 512, .{});
        const offset = blocks.iv.to(.i32).mul(cfg.block_size);
        var block_expert = b.openIfElse(offset.cmpUnsigned(.lt, sums.results[1]), .{zero.type_()});
        // Upper bound of inclusive expert ends handles empty experts without a scan.
        var bounds = b.openWhile(.{ zero, experts }, .{ zero.type_(), zero.type_() });
        bounds.yieldBefore(bounds.before_carried[0].cmpUnsigned(.lt, bounds.before_carried[1]), .{ bounds.before_carried[0], bounds.before_carried[1] });
        const lo = bounds.after_carried[0];
        const hi = bounds.after_carried[1];
        const mid = lo.add(hi).shrU(1);
        const advance = ends.at(mid).cmpUnsigned(.le, offset);
        bounds.yieldAfter(.{ advance.select(mid.add(1), lo), advance.select(hi, mid) });
        block_expert.yieldThen(.{bounds.results[0]});
        block_expert.yieldElse(.{b.constant(.i32, -1)});
        args.expert_ids.set(blocks.iv, block_expert.results[0]);
        blocks.yield(.{});
        fill.yieldElse(.{});
    }
};

const Sort = struct {
    const Result = struct {
        sorted_token_ids: Tensor,
        cumsum: Tensor,
    };

    fn supports(ids: Tensor, sorted_token_ids: Tensor, cumsum: Tensor) bool {
        return ids.rank() == 1 and sorted_token_ids.rank() == 1 and cumsum.rank() == 1 and
            ids.dtype() == .i32 and sorted_token_ids.dtype() == .i32 and cumsum.dtype() == .i32 and
            cumsum.dim(0) >= 2 and cumsum.dim(0) <= std.math.maxInt(i32) and
            ids.dim(0) <= std.math.maxInt(i32) and sorted_token_ids.dim(0) >= ids.dim(0);
    }

    /// cumsum contains one starting offset per expert followed by the total padded
    /// capacity. Each valid route atomically advances its expert's cursor; ordering
    /// within an expert is unspecified. Invalid expert IDs leave both buffers alone.
    fn call(ids: Tensor, sorted_token_ids: Tensor, cumsum: Tensor) Result {
        std.debug.assert(Sort.supports(ids, sorted_token_ids, cumsum));
        if (ids.dim(0) == 0) return .{ .sorted_token_ids = sorted_token_ids, .cumsum = cumsum };
        const routes = ids.dim(0);
        const grid = @min(@divTrunc(routes + 255, 256), 65535);
        const cfg: Config = .{ .routes = routes, .experts = cumsum.dim(0) - 1, .step = grid * 256 };
        const result = Kernel.call(.{ .ids = ids, .sorted = sorted_token_ids, .cumsum_in = cumsum }, .{ .sorted_token_ids = sorted_token_ids.shape(), .cumsum = cumsum.shape() }, .{
            .cfg = cfg,
            .threads = 256,
            .grid = .{ @intCast(grid), 1, 1 },
            .output_operand_aliases = .{ .sorted_token_ids = .sorted, .cumsum = .cumsum_in },
        });
        return .{ .sorted_token_ids = result.sorted_token_ids, .cumsum = result.cumsum };
    }

    const Config = struct {
        routes: i64,
        experts: i64,
        step: i64,
    };

    const Kernel = zml.kernel.fly.Kernel(Config, .{
        .name = "fly_count_and_sort_expert_tokens",
        .inputs = &.{ "ids", "sorted", "cumsum_in" },
        .outputs = &.{ "sorted_token_ids", "cumsum" },
        .run = run,
    });

    fn run(b: *Builder, cfg: Config) fly.FinishError!void {
        const args = Kernel.args(b);
        const start = b.blockId(.x).to(.i64).mul(256).add(b.threadId(.x).to(.i64));
        var routes = b.openFor(start, cfg.routes, cfg.step, .{});
        const id = args.ids.at(routes.iv);
        var valid = b.openIf(id.cmp(.ge, 0).bitAnd(id.cmp(.lt, cfg.experts)));
        const cursor = args.cumsum.emitIter().addOffset(id.to(.i64));
        const rank = b.ptrAtomicAdd(cursor, b.constant(.i32, 1), .agent);
        args.sorted_token_ids.set(rank.to(.i64), routes.iv.to(.i32));
        valid.yieldThen(.{});
        routes.yield(.{});
    }
};
