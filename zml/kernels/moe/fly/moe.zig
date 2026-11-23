const std = @import("std");
const zml = @import("zml/core");
const Tensor = zml.Tensor;
const Shape = zml.Shape;
const fly = zml.kernel.fly;
const Builder = fly.Builder;
const Value = fly.Value;

pub const Align = struct {
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
    pub fn call(ids: Tensor, sorted: Tensor, experts: Tensor, total: Tensor, cumsum: Tensor, block_size: i64) Result {
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
        var fill = b.openIfElse(b.blockId(.x).ne(0), .{});
        {
            var positions = b.openFor(tid64, cfg.capacity, 512, .{});
            {
                args.sorted_token_ids.set(positions.iv, b.constant(.i32, cfg.routes));
                positions.close();
            }
            fill.elseBranch();
        }
        {
            counts.set(tid, zero);
            b.barrier();
            var routes = b.openFor(tid64, cfg.routes, 512, .{});
            {
                const id = args.ids.at(routes.iv);
                var valid_id = b.openIf(id.ge(0).bitAnd(id.lt(cfg.experts)));
                {
                    _ = b.ptrAtomicAdd(counts.emitIter().addOffset(id), one, .workgroup);
                    valid_id.close();
                }
                routes.close();
            }
            b.barrier();

            const padded = counts.at(tid).add(cfg.padding).divUnsigned(cfg.block_size).mul(cfg.block_size);
            const lane = tid.remUnsigned(64);
            const wave = tid.divUnsigned(64);
            var prefix = padded;
            inline for (.{ 1, 2, 4, 8, 16, 32 }) |shift| {
                const peer = prefix.shuffle(.up, shift, 64);
                prefix = prefix.add(lane.cmpUnsigned(.ge, shift).select(peer, zero));
            }
            var last_lane = b.openIf(lane.eq(63));
            {
                waves.set(wave, prefix);
                last_lane.close();
            }
            b.barrier();

            var sums = b.openFor(zero, 8, 1, .{ zero, zero });
            {
                const value = waves.at(sums.iv);
                const preceding = sums.carried[0].add(sums.iv.cmpUnsigned(.lt, wave).select(value, zero));
                const total = sums.carried[1].add(value);
                sums.yield(.{ preceding, total });
            }
            const end = sums.results[0].add(prefix);
            const start = end.sub(padded);
            ends.set(tid, end);
            var has_expert = b.openIf(tid.cmpUnsigned(.lt, cfg.experts));
            {
                args.cumsum.set(tid64, start);
                has_expert.close();
            }
            var first = b.openIf(tid.eq(0));
            {
                args.cumsum.set(b.constant(.i64, cfg.experts), sums.results[1]);
                args.num_tokens_post_pad.set(b.constant(.i64, 0), sums.results[1]);
                first.close();
            }
            b.barrier();

            var blocks = b.openFor(tid64, cfg.blocks, 512, .{});
            {
                const offset = blocks.iv.to(.i32).mul(cfg.block_size);
                var block_expert = b.openIfElse(offset.cmpUnsigned(.lt, sums.results[1]), .{zero});
                {
                    // Upper bound of inclusive expert ends handles empty experts without a scan.
                    var bounds = b.openWhile(.{ zero, experts }, .{ zero, zero });
                    {
                        bounds.yieldBefore(bounds.before_carried[0].cmpUnsigned(.lt, bounds.before_carried[1]), .{ bounds.before_carried[0], bounds.before_carried[1] });
                    }
                    {
                        const lo = bounds.after_carried[0];
                        const hi = bounds.after_carried[1];
                        const mid = lo.add(hi).shrU(1);
                        const advance = ends.at(mid).cmpUnsigned(.le, offset);
                        bounds.yieldAfter(.{ advance.select(mid.add(1), lo), advance.select(hi, mid) });
                    }
                    block_expert.yieldThen(.{bounds.results[0]});
                }
                {
                    block_expert.yieldElse(.{b.constant(.i32, -1)});
                }
                args.expert_ids.set(blocks.iv, block_expert.results[0]);
                blocks.close();
            }
            fill.close();
        }
    }
};

pub const Sort = struct {
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
    pub fn call(ids: Tensor, sorted_token_ids: Tensor, cumsum: Tensor) Result {
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
        {
            const id = args.ids.at(routes.iv);
            var valid = b.openIf(id.ge(0).bitAnd(id.lt(cfg.experts)));
            {
                const cursor = args.cumsum.emitIter().addOffset(id.to(.i64));
                const rank = b.ptrAtomicAdd(cursor, b.constant(.i32, 1), .agent);
                args.sorted_token_ids.set(rank.to(.i64), routes.iv.to(.i32));
                valid.close();
            }
            routes.close();
        }
    }
};

pub const Mfma = struct {
    /// MFMA loads full 64-column by 128-reduction tiles without tail masks.
    pub fn supportsShape(n: i64, k: i64) bool {
        return n > 0 and k > 0 and @mod(n, 64) == 0 and @mod(k, 128) == 0;
    }

    /// The caller owns route localization, activation and reduction.
    /// `expert_ids` contains local bank indices, including -1 for remote experts.
    /// Sorted routes must contain a real prefix followed by padding per expert.
    /// `block_m` is their alignment; native tiles contain 16 or 32 rows.
    /// `scales` holds one entry per 32 weights: an E8M0 byte (MXFP4), or an i16
    /// pairing two E4M3FN bytes that each cover 16 weights (NVFP4).
    pub fn call(
        input: Tensor,
        weight: Tensor,
        scales: Tensor,
        sorted_ids: Tensor,
        expert_ids: Tensor,
        padded_count: Tensor,
        output: zml.Shape,
        topk: usize,
        naive: bool,
        block_m: usize,
    ) Tensor {
        std.debug.assert(input.rank() == 2 and input.dtype() == .bf16 and weight.rank() == 3 and
            (weight.dtype() == .u8 or weight.dtype() == .i8) and scales.rank() == 3 and (scales.dtype() == .u8 or scales.dtype() == .i16) and
            sorted_ids.rank() == 1 and sorted_ids.dtype() == .i32 and expert_ids.rank() == 1 and expert_ids.dtype() == .i32 and
            padded_count.count() == 1 and padded_count.dtype() == .i32 and output.rank() >= 2 and output.dtype() == .bf16 and
            topk > 0 and block_m > 0 and block_m % 16 == 0);
        const n = weight.dim(1);
        const k = input.dim(1);
        std.debug.assert(supportsShape(n, k));
        std.debug.assert(input.dim(0) > 0 and weight.dim(0) > 0 and weight.dim(2) == @divExact(k, 2) and
            scales.dim(0) == weight.dim(0) and scales.dim(1) == n and scales.dim(2) == @divExact(k, 32) and
            output.dim(-1) == n);
        // Keep packed-word counts and route/output indices within signed i32.
        std.debug.assert(weight.count() / 4 <= std.math.maxInt(i32) and output.count() <= std.math.maxInt(i32) and
            sorted_ids.count() <= std.math.maxInt(i32));
        const routes: i64 = @intCast(output.count() / @as(usize, @intCast(n)));
        std.debug.assert(topk <= routes and @mod(@as(usize, @intCast(routes)), topk) == 0 and
            input.dim(0) == @divTrunc(routes, @as(i64, @intCast(topk))));
        const capacity: i64 = @intCast(sorted_ids.count());
        const blocks: i64 = @intCast(expert_ids.count());
        if (naive) {
            std.debug.assert(blocks == routes);
        } else {
            std.debug.assert(capacity > 0 and block_m <= capacity and @mod(capacity, @as(i64, @intCast(block_m))) == 0 and
                blocks == @divExact(capacity, @as(i64, @intCast(block_m))));
        }
        if (naive) return Gemv.call(input, weight, scales, expert_ids, output, topk);
        return callMfma(input, weight, scales, sorted_ids, expert_ids, padded_count, output, topk, naive, block_m);
    }

    fn callMfma(input: Tensor, weight: Tensor, scales: Tensor, sorted_ids: Tensor, expert_ids: Tensor, padded_count: Tensor, output: zml.Shape, topk: usize, naive: bool, block_m: usize) Tensor {
        const n = weight.dim(1);
        const k = input.dim(1);
        const routes = input.dim(0) * @as(i64, @intCast(topk));
        const native_m: i64 = if (!naive and block_m % 32 == 0) 32 else 16;
        const cfg: Config = .{
            .tokens = input.dim(0),
            .experts = weight.dim(0),
            .n = n,
            .k = k,
            .ka = @divExact(k, 4),
            .kb = @divExact(k, 8),
            .ks = @divExact(k, 32),
            .topk = @as(i64, @intCast(topk)),
            .routes = routes,
            .capacity = @as(i64, @intCast(sorted_ids.count())),
            .blocks = @as(i64, @intCast(expert_ids.count())),
            .subtiles = @divExact(@as(i64, @intCast(block_m)), native_m),
            .native_m = native_m,
            .wide = @as(i64, @intFromBool(native_m == 32)),
            .stage_elements = native_m * 128,
            .shared_elements = native_m * 256,
            .naive = @as(i64, @intFromBool(naive)),
            .nvfp4 = @as(i64, @intFromBool(scales.dtype() == .i16)),
        };
        const a = input.reshape(.{ input.dim(0), @divExact(k, 4), 4 }).bitCast(.i64);
        const b = weight.reshape(.{ weight.dim(0), n, @divExact(k, 8), 4 }).bitCast(.i32);
        return Kernel.call(.{ .a = a, .b = b, .scales = scales, .sorted = sorted_ids, .experts = expert_ids, .padded = padded_count.reshape(.{1}) }, .{
            .output = zml.Shape.init(.{ routes, n }, .bf16),
        }, .{
            .cfg = cfg,
            .threads = 256,
            .grid = .{ @intCast(@divExact(n, 64)), @intCast(if (naive) routes else @divExact(cfg.capacity, native_m)), 1 },
        }).output.reshape(output);
    }

    const Config = struct {
        tokens: i64,
        experts: i64,
        n: i64,
        k: i64,
        ka: i64,
        kb: i64,
        ks: i64,
        topk: i64,
        routes: i64,
        capacity: i64,
        blocks: i64,
        subtiles: i64,
        native_m: i64,
        wide: i64,
        stage_elements: i64,
        shared_elements: i64,
        naive: i64,
        nvfp4: i64,
    };

    const Kernel = zml.kernel.fly.Kernel(Config, .{
        .name = "mxfp4_raw_mfma",
        .inputs = &.{ "a", "b", "scales", "sorted", "experts", "padded" },
        .outputs = &.{"output"},
        .run = run,
    });

    fn run(b: *Builder, cfg: Config) fly.FinishError!void {
        const args = Kernel.args(b);
        const zero = b.constant(.i32, 0);
        const fzero = b.constant(.f32, 0.0);
        const zeros = fzero.splat(4);
        const block_n = b.blockId(.x).to(.i64);
        const block_m = b.blockId(.y).to(.i64);
        const route_start = block_m.mul(cfg.native_m);
        const has_rows = if (cfg.naive != 0) b.constant(.i1, 1) else blk: {
            const padded = args.padded.at(b.constant(.i64, 0)).to(.i64);
            var has_block = b.openIfElse(route_start.cmpUnsigned(.lt, padded), .{.i1});
            {
                has_block.yieldThen(.{args.sorted.at(route_start).ne(cfg.routes)});
            }
            {
                has_block.yieldElse(.{b.constant(.i1, 0)});
            }
            break :blk has_block.results[0];
        };
        var block = b.openIf(has_rows);
        {
            const expert_block = if (cfg.naive != 0) block_m else block_m.divUnsigned(cfg.subtiles);
            const eid = args.experts.at(expert_block);
            const valid_expert = eid.ge(0).bitAnd(eid.lt(cfg.experts));
            const thread = b.threadId(.x).to(.i64);
            const lane = thread.remUnsigned(64);
            const wave = thread.divUnsigned(64);
            const mn_lane = lane.remUnsigned(16);
            const k_lane = lane.divUnsigned(16);
            const out_n = block_n.mul(64).add(wave.mul(16)).add(mn_lane);
            // Each expert has a contiguous real prefix before route-count padding.
            const second_active = if (cfg.wide != 0) blk: {
                const first = args.sorted.at(route_start.add(16));
                break :blk first.ge(0).bitAnd(first.lt(cfg.routes));
            } else b.constant(.i1, 0);

            var sums = b.openIfElse(valid_expert, .{ zeros, zeros });
            {
                const expert = eid.to(.i64);
                const load_row = thread.divUnsigned(16);
                const load_chunk = thread.remUnsigned(16);
                const a_route = routeForRow(b, args.sorted, block_m, route_start, load_row, cfg.naive != 0);
                const a_valid = a_route.ge(0).bitAnd(a_route.lt(cfg.routes));
                const a_base = a_route.to(.i64).divUnsigned(cfg.topk).mul(cfg.ka);
                var second_route = b.openIfElse(second_active, .{zero});
                {
                    second_route.yieldThen(.{args.sorted.at(route_start.add(load_row).add(16))});
                }
                {
                    second_route.yieldElse(.{b.constant(.i32, -1)});
                }
                const a_route_g1 = second_route.results[0];
                const a_valid_g1 = a_route_g1.ge(0).bitAnd(a_route_g1.lt(cfg.routes));
                const a_base_g1 = a_route_g1.to(.i64).divUnsigned(cfg.topk).mul(cfg.ka);

                // One allocation, two disjoint ping-pong regions; never two identical leaves.
                const shared = b.sharedArray(.i16, cfg.shared_elements, 16);
                const load_a = b.copyAtom(.{ .universal = 128 }, .i64);
                const load_b = b.copyAtom(.{ .universal = 128 }, .i32);
                const copy_shared = b.copyAtom(.{ .universal = 128 }, .i16);
                const loader: TileLoader = .{
                    .a = args.a,
                    .weight = args.b,
                    .scales = args.scales,
                    .shared = shared,
                    .load_a = load_a,
                    .load_b = load_b,
                    .copy_shared = copy_shared,
                    .a_base = a_base,
                    .a_base_g1 = a_base_g1,
                    .a_valid = a_valid,
                    .a_valid_g1 = a_valid_g1,
                    .second_active = second_active,
                    .a_col = load_chunk.mul(2),
                    .a_store = load_row.mul(128).add(load_chunk.bitXor(load_row).mul(8)),
                    .b_base = expert.mul(cfg.n * cfg.kb).add(out_n.mul(cfg.kb)),
                    .expert = expert,
                    .out_n = out_n,
                    .k_lane = k_lane,
                    .nvfp4 = cfg.nvfp4 != 0,
                };
                const initial = loader.load(b, b.constant(.i64, 0), b.constant(.i64, 0));
                b.barrier();
                const atom = b.mmaAtom((fly.fly_rocdl.MmaOpCDNA3MFMAType.get(b.ctx, .{
                    .m = 16,
                    .n = 16,
                    .k = 16,
                    .elemTyA = fly.DType.bf16.toMlir(b.ctx),
                    .elemTyB = fly.DType.bf16.toMlir(b.ctx),
                    .elemTyAcc = fly.DType.f32.toMlir(b.ctx),
                }) catch return error.InvalidMlir).type_());
                var k = b.openFor(b.constant(.i64, 0), cfg.k, 128, .{ zeros, zeros, initial.words, initial.scale });
                {
                    const read_base = k.iv.divUnsigned(128).bitAnd(1).mul(cfg.stage_elements);
                    const write_base = k.iv.divUnsigned(128).bitAnd(1).bitXor(1).mul(cfg.stage_elements);
                    var a_fragments: [4]Value = undefined;
                    var a_fragments_g1: [4]Value = undefined;
                    inline for (0..4) |u| {
                        const offset = read_base.add(mn_lane.mul(128)).add(k_lane.mul(4).add(u).bitXor(mn_lane).mul(8));
                        a_fragments[u] = b.copyAtomLoad(copy_shared, vectorView(shared, offset, 8));
                    }
                    inline for (0..4) |u| {
                        const offset = read_base.add(mn_lane.mul(128)).add(k_lane.mul(4).add(u).bitXor(mn_lane).mul(8)).add(2048);
                        const azero = b.constant(.i16, 0).splat(8);
                        var second = b.openIfElse(second_active, .{azero});
                        {
                            second.yieldThen(.{b.copyAtomLoad(copy_shared, vectorView(shared, offset, 8))});
                        }
                        {
                            second.yieldElse(.{azero});
                        }
                        a_fragments_g1[u] = second.results[0];
                    }
                    // Prefetch the other LDS region only after all current A fragments are read.
                    const next_k = k.iv.add(128);
                    var next = b.openIfElse(next_k.cmpUnsigned(.lt, cfg.k), .{ initial.words, initial.scale });
                    {
                        const next_tile = loader.load(b, next_k, write_base);
                        next.yieldThen(.{ next_tile.words, next_tile.scale });
                    }
                    {
                        next.yieldElse(.{ b.constant(.i32, 0).splat(4), if (cfg.nvfp4 != 0) fzero.splat(2) else fzero });
                    }
                    var acc = k.carried[0];
                    var acc_g1 = k.carried[1];
                    inline for (0..4) |u| {
                        // A lane's four words share one MXFP4 scale, or two NVFP4 scales by halves.
                        const scale = if (cfg.nvfp4 != 0) k.carried[3].extract(u / 2) else k.carried[3];
                        const decoded = decodeWeight(b, k.carried[2].extract(u), scale);
                        const a0 = a_fragments[u].shuffleVector(a_fragments[u], &.{ 0, 1, 2, 3 }).bitCast(.bf16);
                        const a1 = a_fragments[u].shuffleVector(a_fragments[u], &.{ 4, 5, 6, 7 }).bitCast(.bf16);
                        const w0 = decoded.shuffleVector(decoded, &.{ 0, 1, 2, 3 });
                        const w1 = decoded.shuffleVector(decoded, &.{ 4, 5, 6, 7 });
                        acc = b.mmaAtomCall(atom, a0, w0, acc);
                        acc = b.mmaAtomCall(atom, a1, w1, acc);
                        var second = b.openIfElse(second_active, .{zeros});
                        {
                            const a0_g1 = a_fragments_g1[u].shuffleVector(a_fragments_g1[u], &.{ 0, 1, 2, 3 }).bitCast(.bf16);
                            const a1_g1 = a_fragments_g1[u].shuffleVector(a_fragments_g1[u], &.{ 4, 5, 6, 7 }).bitCast(.bf16);
                            const first_acc = b.mmaAtomCall(atom, a0_g1, w0, acc_g1);
                            second.yieldThen(.{b.mmaAtomCall(atom, a1_g1, w1, first_acc)});
                        }
                        {
                            second.yieldElse(.{acc_g1});
                        }
                        acc_g1 = second.results[0];
                    }
                    // All current reads finish before a later iteration reuses this LDS region.
                    b.barrier();
                    k.yield(.{ acc, acc_g1, next.results[0], next.results[1] });
                }
                sums.yieldThen(.{ k.results[0], k.results[1] });
            }
            {
                sums.yieldElse(.{ zeros, zeros });
            }

            inline for (0..4) |i| {
                storeRow(b, args.output, args.sorted, block_m, route_start, k_lane.mul(4).add(i), out_n, sums.results[0].extract(i), cfg);
            }
            var store_second = b.openIf(second_active);
            {
                inline for (0..4) |i| {
                    storeRow(b, args.output, args.sorted, block_m, route_start, k_lane.mul(4).add(i + 16), out_n, sums.results[1].extract(i), cfg);
                }
                store_second.close();
            }
            block.close();
        }
    }

    fn routeForRow(b: *Builder, sorted: Value, block: Value, start: Value, row: Value, naive: bool) Value {
        return if (naive) row.eq(0).select(block.to(.i32), b.constant(.i32, -1)) else sorted.at(start.add(row));
    }

    fn storeRow(b: *Builder, output: Value, sorted: Value, block: Value, start: Value, row: Value, col: Value, value: Value, cfg: Config) void {
        const route = routeForRow(b, sorted, block, start, row, cfg.naive != 0);
        var valid = b.openIf(route.ge(0).bitAnd(route.lt(cfg.routes)));
        {
            output.set(.{ route.to(.i64), col }, value.to(.bf16));
            valid.close();
        }
    }

    fn vectorView(tensor: Value, offset: Value, comptime n: i64) Value {
        return tensor.emitIter().addOffset(offset).view(fly.L(n, 1));
    }

    const TileLoader = struct {
        a: Value,
        weight: Value,
        scales: Value,
        shared: Value,
        load_a: Value,
        load_b: Value,
        copy_shared: Value,
        a_base: Value,
        a_base_g1: Value,
        a_valid: Value,
        a_valid_g1: Value,
        second_active: Value,
        a_col: Value,
        a_store: Value,
        b_base: Value,
        expert: Value,
        out_n: Value,
        k_lane: Value,
        nvfp4: bool,

        fn load(self: TileLoader, b: *Builder, k: Value, write_base: Value) struct { words: Value, scale: Value } {
            self.storeActivation(b, self.a_base, self.a_valid, k, self.a_store.add(write_base));
            var second = b.openIf(self.second_active);
            {
                self.storeActivation(b, self.a_base_g1, self.a_valid_g1, k, self.a_store.add(write_base).add(2048));
                second.close();
            }
            const offset = self.b_base.add(k.divUnsigned(8)).add(self.k_lane.mul(4));
            const words = b.copyAtomLoad(self.load_b, vectorView(self.weight, offset, 4));
            const scale = self.scales.at(.{ self.expert, self.out_n, k.divUnsigned(32).add(self.k_lane) });
            return .{ .words = words, .scale = if (self.nvfp4) decodeE4m3Pair(b, scale) else decodeE8m0(b, scale) };
        }

        fn storeActivation(self: TileLoader, b: *Builder, base: Value, valid: Value, k: Value, destination: Value) void {
            const zeros = b.constant(.i64, 0).splat(2);
            var guarded = b.openIfElse(valid, .{zeros});
            {
                const offset = base.add(k.divUnsigned(4)).add(self.a_col);
                guarded.yieldThen(.{b.copyAtomLoad(self.load_a, vectorView(self.a, offset, 2))});
            }
            {
                guarded.yieldElse(.{zeros});
            }
            b.copyAtomStore(self.copy_shared, guarded.results[0].bitCast(.i16), vectorView(self.shared, destination, 8));
        }
    };

    fn decodeWeight(b: *Builder, word: Value, scale: Value) Value {
        const lo = b.constant(.i32, 0x44403800);
        const hi = b.constant(.i32, 0x54504c48);
        const even = b.perm(hi, lo, word.bitAnd(0x07070707));
        const odd = b.perm(hi, lo, word.shrU(4).bitAnd(0x07070707));
        const ev0 = b.cvtPkF32Fp8(even, false);
        const od0 = b.cvtPkF32Fp8(odd, false);
        const ev1 = b.cvtPkF32Fp8(even, true);
        const od1 = b.cvtPkF32Fp8(odd, true);
        const first = ev0.shuffleVector(od0, &.{ 0, 2, 1, 3 });
        const last = ev1.shuffleVector(od1, &.{ 0, 2, 1, 3 });
        const magnitude = first.shuffleVector(last, &.{ 0, 1, 2, 3, 4, 5, 6, 7 });
        var shifts = b.constant(.i32, 0).splat(8);
        inline for (0..8) |i| shifts = shifts.insert(i, b.constant(.i32, i * 4));
        const signs = word.splat(8).shrU(shifts).bitAnd(8).shl(28);
        // FNUZ has no -0: restore FP4's sign in F32 before scaling and BF16 rounding.
        return magnitude.bitCast(.i32).bitOr(signs).bitCast(.f32).mul(scale).to(.bf16);
    }
};

/// Routed GEMV for decode-sized batches. A block owns one row tile of an expert
/// and up to `group` of the routes that select it, so each weight row is read
/// once per group. Lanes stream 16-byte chunks of 32 weights along K; routes
/// find their group by scanning the expert IDs, without a sorting pass.
const Gemv = struct {
    const group = 8;
    const threads = 256;

    const Config = struct {
        n: i64,
        experts: i64,
        routes: i64,
        topk: i64,
        x_words: i64,
        k_words: i64,
        chunks: i64,
        /// Lanes sharing one weight row, and the chunks each lane loads at once.
        lanes: i64,
        batch: i64,
        batches: i64,
        rows: i64,
        nvfp4: i64,
    };

    /// `ids` holds one local expert per route, negative or out of bank for remote
    /// experts, whose output rows are zero. Route `r` reads input row `r / topk`.
    pub fn call(x: Tensor, weight: Tensor, scales: Tensor, ids: Tensor, output: zml.Shape, topk: usize) Tensor {
        const n = weight.dim(1);
        const k = x.dim(1);
        const routes: i64 = @intCast(ids.count());
        std.debug.assert(x.dtype() == .bf16 and (weight.dtype() == .u8 or weight.dtype() == .i8) and
            weight.dim(2) * 2 == k and @mod(k, 32) == 0 and routes == x.dim(0) * @as(i64, @intCast(topk)) and routes <= threads);
        std.debug.assert(scales.dim(0) == weight.dim(0) and scales.dim(1) == n and scales.dim(2) == @divExact(k, 32) and
            (scales.dtype() == .u8 or scales.dtype() == .i16));
        const chunks = @divExact(k, 32);
        // Enough lanes per row that each loads at most 12 chunks in one batch.
        var lanes: i64 = 4;
        while (lanes < 16 and std.math.divCeil(i64, chunks, lanes) catch unreachable > 12) lanes *= 2;
        const per_lane = std.math.divCeil(i64, chunks, lanes) catch unreachable;
        const batches = std.math.divCeil(i64, per_lane, 12) catch unreachable;
        const cfg: Config = .{
            .n = n,
            .experts = weight.dim(0),
            .routes = routes,
            .topk = @intCast(topk),
            .x_words = @divExact(k, 2),
            .k_words = @divExact(k, 8),
            .chunks = chunks,
            .lanes = lanes,
            .batch = std.math.divCeil(i64, per_lane, batches) catch unreachable,
            .batches = batches,
            .rows = @divExact(threads, lanes),
            .nvfp4 = @intFromBool(scales.dtype() == .i16),
        };
        const a = x.reshape(.{ x.dim(0), cfg.x_words, 2 }).bitCast(.i32);
        const b = weight.reshape(.{ weight.dim(0), n, cfg.k_words, 4 }).bitCast(.i32);
        return Kernel.call(.{ .a = a, .b = b, .scales = scales, .ids = ids.reshape(.{routes}) }, .{
            .output = zml.Shape.init(.{ routes, n }, .bf16),
        }, .{
            .cfg = cfg,
            .threads = threads,
            .grid = .{ @intCast(std.math.divCeil(i64, n, cfg.rows) catch unreachable), @intCast(routes), 1 },
        }).output.reshape(output);
    }

    const Kernel = zml.kernel.fly.Kernel(Config, .{
        .name = "fp4_grouped_gemv",
        .inputs = &.{ "a", "b", "scales", "ids" },
        .outputs = &.{"output"},
        .run = run,
    });

    fn run(b: *Builder, cfg: Config) fly.FinishError!void {
        const args = Kernel.args(b);
        const zero = b.constant(.i32, 0);
        const one = b.constant(.i32, 1);
        const fzero = b.constant(.f32, 0.0);
        const route = b.blockId(.y).to(.i64);
        const thread32 = b.threadId(.x);
        const thread = thread32.to(.i64);
        const lane = thread.remUnsigned(cfg.lanes);
        const row = b.blockId(.x).to(.i64).mul(cfg.rows).add(thread.divUnsigned(cfg.lanes));
        const in_rows = row.cmpUnsigned(.lt, cfg.n);
        const expert32 = args.ids.at(route);
        const local = expert32.ge(0).bitAnd(expert32.lt(cfg.experts));

        var remote = b.openIfElse(local, .{});
        {
            // Thread t ranks route t among the routes of this expert; each run of
            // `group` consecutive ranks is one group, owned by its first route.
            const ranks = b.sharedArray(.i32, threads, 16);
            const waves = b.sharedArray(.i32, threads / 64, 16);
            const slots = b.sharedArray(.i32, group, 16);
            var routed = b.openIfElse(thread.cmpUnsigned(.lt, cfg.routes), .{zero});
            {
                routed.yieldThen(.{args.ids.at(thread).eq(expert32).select(one, zero)});
            }
            {
                routed.yieldElse(.{zero});
            }
            const same = routed.results[0];
            const wave_lane = thread32.remUnsigned(64);
            const wave = thread32.divUnsigned(64);
            var inclusive = same;
            inline for (.{ 1, 2, 4, 8, 16, 32 }) |shift| {
                const peer = inclusive.shuffle(.up, shift, 64);
                inclusive = inclusive.add(wave_lane.cmpUnsigned(.ge, shift).select(peer, zero));
            }
            var last_lane = b.openIf(wave_lane.eq(63));
            {
                waves.set(wave, inclusive);
                last_lane.close();
            }
            b.barrier();
            var totals = b.openFor(zero, threads / 64, 1, .{ zero, zero });
            {
                const value = waves.at(totals.iv);
                totals.yield(.{ totals.carried[0].add(totals.iv.lt(wave).select(value, zero)), totals.carried[1].add(value) });
            }
            const rank = totals.results[0].add(inclusive).sub(same);
            ranks.set(thread32, rank);
            b.barrier();
            const first = ranks.at(route.to(.i32));
            const remaining = totals.results[1].sub(first);
            const count = remaining.lt(group).select(remaining, b.constant(.i32, group));
            var leader = b.openIf(first.remUnsigned(group).eq(0));
            {
                const offset = rank.sub(first);
                var member = b.openIf(same.eq(1).bitAnd(offset.ge(0)).bitAnd(offset.lt(group)));
                {
                    slots.set(offset, thread32);
                    member.close();
                }
                b.barrier();
                var members: [group]Value = undefined;
                var a_base: [group]Value = undefined;
                inline for (0..group) |m| {
                    members[m] = count.gt(m).select(slots.at(b.constant(.i32, m)), zero).to(.i64);
                    a_base[m] = members[m].divUnsigned(cfg.topk).mul(cfg.x_words);
                }
                const sums = accumulate(b, args, cfg, expert32.to(.i64), row, lane, count, a_base);
                var store = b.openIf(lane.eq(0).bitAnd(in_rows));
                {
                    inline for (0..group) |m| {
                        var write = b.openIf(count.gt(m));
                        {
                            args.output.set(.{ members[m], row }, sums[m].to(.bf16));
                            write.close();
                        }
                    }
                    store.close();
                }
                leader.close();
            }
            remote.elseBranch();
        }
        {
            var store = b.openIf(lane.eq(0).bitAnd(in_rows));
            {
                args.output.set(.{ route, row }, fzero.to(.bf16));
                store.close();
            }
            remote.close();
        }
    }

    /// Each lane's partial dot products with its group's input rows, summed over the lanes of its weight row.
    fn accumulate(b: *Builder, args: Kernel.Args, cfg: Config, expert: Value, row: Value, lane: Value, count: Value, a_base: [group]Value) [group]Value {
        const fzero = b.constant(.f32, 0.0);
        const load = b.copyAtom(.{ .universal = 128 }, .i32);
        // Rows past N read row 0 and are never stored.
        const b_base = expert.mul(cfg.n * cfg.k_words).add(row.cmpUnsigned(.lt, cfg.n).select(row, b.constant(.i64, 0)).mul(cfg.k_words));
        const scale_row = row.cmpUnsigned(.lt, cfg.n).select(row, b.constant(.i64, 0));
        var inits: [group]Value = undefined;
        for (&inits) |*acc| acc.* = fzero;
        var batches = b.openFor(b.constant(.i64, 0), cfg.batches, 1, inits);
        {
            // Unconditional loads keep a batch in flight together; chunks past K
            // reread the last chunk and are zeroed through their scales.
            var words: [12]Value = undefined;
            var scales: [12]Value = undefined;
            var chunks: [12]Value = undefined;
            for (0..@intCast(cfg.batch)) |j| {
                chunks[j] = batches.iv.mul(cfg.batch).add(@as(i64, @intCast(j))).mul(cfg.lanes).add(lane);
                const chunk = chunks[j].cmpUnsigned(.lt, cfg.chunks).select(chunks[j], b.constant(.i64, cfg.chunks - 1));
                words[j] = b.copyAtomLoad(load, args.b.emitIter().addOffset(b_base.add(chunk.mul(4))).view(fly.L(4, 1)));
                scales[j] = args.scales.at(.{ expert, scale_row, chunk });
            }
            var acc: [group]Value = batches.carried;
            for (0..@intCast(cfg.batch)) |j| {
                const chunk = chunks[j].cmpUnsigned(.lt, cfg.chunks).select(chunks[j], b.constant(.i64, cfg.chunks - 1));
                var weights: [4][4]Value = undefined;
                inline for (0..4) |u| weights[u] = decodeWord(b, words[j].extract(u));
                const decoded = if (cfg.nvfp4 != 0) decodeE4m3Pair(b, scales[j]) else decodeE8m0(b, scales[j]).splat(2);
                const block_scales = chunks[j].cmpUnsigned(.lt, cfg.chunks).select(decoded, fzero.splat(2));
                inline for (0..group) |m| {
                    var member = b.openIfElse(count.gt(m), .{.f32});
                    {
                        member.yieldThen(.{dot(b, args, load, a_base[m], chunk, weights, block_scales, acc[m])});
                    }
                    {
                        member.yieldElse(.{acc[m]});
                    }
                    acc[m] = member.results[0];
                }
            }
            batches.yield(acc);
        }
        var sums = batches.results;
        inline for (0..group) |m| {
            var shift: i32 = 1;
            while (shift < cfg.lanes) : (shift *= 2) sums[m] = sums[m].add(sums[m].shuffleXor(shift, 64));
        }
        return sums;
    }

    /// `acc` plus one input row's dot product with a decoded 32-weight chunk.
    fn dot(b: *Builder, args: Kernel.Args, load: Value, a_base: Value, chunk: Value, weights: [4][4]Value, block_scales: Value, acc: Value) Value {
        const fzero = b.constant(.f32, 0.0);
        var partial: [2]Value = .{ fzero, fzero };
        inline for (0..4) |q| {
            const xs = b.copyAtomLoad(load, args.a.emitIter().addOffset(a_base.add(chunk.mul(16)).add(q * 4)).view(fly.L(4, 1)));
            inline for (0..4) |pair| {
                const x = xs.extract(pair);
                const lo = x.shl(16).bitCast(.f32);
                const hi = x.bitAnd(@as(i32, @bitCast(@as(u32, 0xffff0000)))).bitCast(.f32);
                // Word q holds weights 8q..8q+7; `pair` selects two of them.
                const w = weights[q];
                partial[q / 2] = b.fma(lo, w[(pair / 2) * 2].extract(pair % 2), b.fma(hi, w[(pair / 2) * 2 + 1].extract(pair % 2), partial[q / 2]));
            }
        }
        return b.fma(partial[0], block_scales.extract(0), b.fma(partial[1], block_scales.extract(1), acc));
    }

    /// Eight FP4 weights as F32 pairs: (0, 2), (1, 3), (4, 6), (5, 7).
    fn decodeWord(b: *Builder, word: Value) [4]Value {
        const even = b.perm(b.constant(.i32, 0xd4d0ccc8), b.constant(.i32, 0xc4c0b800), word.bitAnd(0x07070707))
            .bitAnd(word.shl(4).bitOr(0x7f7f7f7f));
        const odd = b.perm(b.constant(.i32, 0xd4d0ccc8), b.constant(.i32, 0xc4c0b800), word.shrU(4).bitAnd(0x07070707))
            .bitAnd(word.bitOr(0x7f7f7f7f));
        return .{ b.cvtPkF32Fp8(even, false), b.cvtPkF32Fp8(odd, false), b.cvtPkF32Fp8(even, true), b.cvtPkF32Fp8(odd, true) };
    }
};

/// NVFP4 rounding of BF16 activation rows as `quantization.quantizeNvfp4Rows`,
/// widened back to BF16, which represents the rounded values exactly. A block
/// rounds one row; each thread rounds one 16-value block of it.
pub const RoundNvfp4 = struct {
    const threads = 256;

    const Config = struct {
        words: i64,
        blocks: i64,
        four_or_six: i64,
    };

    pub fn call(x: Tensor, four_or_six: bool) Tensor {
        std.debug.assert(x.rank() == 2 and x.dtype() == .bf16 and @mod(x.dim(1), 16) == 0 and x.dim(1) <= 16 * threads);
        const cfg: Config = .{
            .words = @divExact(x.dim(1), 2),
            .blocks = @divExact(x.dim(1), 16),
            .four_or_six = @intFromBool(four_or_six),
        };
        const words = x.reshape(.{ x.dim(0), cfg.words, 2 }).bitCast(.i32);
        return Kernel.call(.{ .x = words }, .{ .output = words.shape() }, .{
            .cfg = cfg,
            .threads = threads,
            .grid = .{ @intCast(x.dim(0)), 1, 1 },
        }).output.bitCast(.bf16).reshape(x.shape());
    }

    const Kernel = zml.kernel.fly.Kernel(Config, .{
        .name = "round_nvfp4",
        .inputs = &.{"x"},
        .outputs = &.{"output"},
        .run = run,
    });

    fn run(b: *Builder, cfg: Config) fly.FinishError!void {
        const args = Kernel.args(b);
        const fzero = b.constant(.f32, 0.0);
        const thread32 = b.threadId(.x);
        const block = thread32.to(.i64);
        const valid = block.cmpUnsigned(.lt, cfg.blocks);
        const offset = b.blockId(.x).to(.i64).mul(cfg.words).add(valid.select(block, b.constant(.i64, 0)).mul(8));
        const access = b.copyAtom(.{ .universal = 128 }, .i32);
        var values: [16]Value = undefined;
        inline for (0..2) |h| {
            const words = b.copyAtomLoad(access, args.x.emitIter().addOffset(offset.add(h * 4)).view(fly.L(4, 1)));
            inline for (0..4) |w| {
                const word = words.extract(w);
                values[h * 8 + w * 2] = word.shl(16).bitCast(.f32);
                values[h * 8 + w * 2 + 1] = word.bitAnd(@as(i32, @bitCast(@as(u32, 0xffff0000)))).bitCast(.f32);
            }
        }
        var block_max = fzero;
        inline for (values) |v| block_max = block_max.maximum(abs(v));
        block_max = valid.select(block_max, fzero);

        const maxima = b.sharedArray(.f32, threads / 64, 16);
        var row_max = block_max;
        inline for (.{ 32, 16, 8, 4, 2, 1 }) |shift| row_max = row_max.maximum(row_max.shuffleXor(shift, 64));
        var first_lane = b.openIf(thread32.remUnsigned(64).eq(0));
        {
            maxima.set(thread32.divUnsigned(64), row_max);
            first_lane.close();
        }
        b.barrier();
        inline for (0..threads / 64) |w| row_max = row_max.maximum(maxima.at(b.constant(.i32, w)));

        var store = b.openIf(valid);
        {
            // The smallest power of two at least row_max / 1536 keeps block scales in E4M3's normal range.
            const scaled_max = row_max.mul(1.0 / 1536.0).maximum(std.math.floatMin(f32)).bitCast(.i32);
            const row_scale = scaled_max.add(0x7fffff).bitAnd(@as(i32, @bitCast(@as(u32, 0xff800000)))).bitCast(.f32);
            var magnitudes: [16]Value = undefined;
            inline for (&magnitudes, values) |*m, v| m.* = abs(v).div(row_scale);
            const normalized_max = block_max.div(row_scale);
            const six = candidate(b, magnitudes, normalized_max, 6.0);
            var chosen = six;
            if (cfg.four_or_six != 0) {
                // Ties keep amax / 6.
                const four = candidate(b, magnitudes, normalized_max, 4.0);
                const use_four = four.err.lt(six.err);
                inline for (&chosen.values, four.values) |*c, f| c.* = use_four.select(f, c.*);
            }
            inline for (0..2) |h| {
                var words = b.constant(.i32, 0).splat(4);
                inline for (0..4) |w| {
                    var halves: [2]Value = undefined;
                    inline for (&halves, 0..) |*half, j| {
                        const i = h * 8 + w * 2 + j;
                        const sign = values[i].bitCast(.i32).bitAnd(@as(i32, @bitCast(@as(u32, 0x80000000))));
                        half.* = chosen.values[i].mul(row_scale).bitCast(.i32).bitOr(sign);
                    }
                    words = words.insert(w, halves[0].shrU(16).bitOr(halves[1].bitAnd(@as(i32, @bitCast(@as(u32, 0xffff0000))))));
                }
                b.copyAtomStore(access, words, args.output.emitIter().addOffset(offset.add(h * 4)).view(fly.L(4, 1)));
            }
            store.close();
        }
    }

    /// Rounded magnitudes of one block for an E4M3 scale of block_max / `max`, and their absolute error.
    const Candidate = struct { values: [16]Value, err: Value };

    fn candidate(b: *Builder, magnitudes: [16]Value, block_max: Value, comptime max: f32) Candidate {
        const clamped = block_max.mul(1.0 / max).maximum(0x1p-6);
        const raw = clamped.gt(448.0).select(b.constant(.f32, 448.0), clamped);
        // Round to E4M3's three mantissa bits, to nearest even; `raw` is a normal E4M3 magnitude.
        const bits = raw.bitCast(.i32);
        const scale = bits.add(0x7ffff).add(bits.shrU(20).bitAnd(1)).bitAnd(@as(i32, @bitCast(@as(u32, 0xfff00000)))).bitCast(.f32);
        // E2M1 rounds to nearest, ties to even, saturating at 6. Each threshold and
        // value times the scale is exact, and a BF16 magnitude never falls within
        // half an ulp of a scaled threshold, so comparing the magnitude against
        // scaled thresholds decides as rounding magnitude / scale would.
        var thresholds: [7]Value = undefined;
        var levels: [8]Value = undefined;
        inline for (&thresholds, .{ 5.0, 3.5, 2.5, 1.75, 1.25, 0.75, 0.25 }) |*t, value| t.* = scale.mul(value);
        inline for (&levels, .{ 6.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.0 }) |*l, value| l.* = scale.mul(value);
        var result: Candidate = .{ .values = undefined, .err = b.constant(.f32, 0.0) };
        inline for (&result.values, magnitudes) |*q, m| {
            q.* = levels[0];
            inline for (thresholds, levels[1..], 0..) |t, level, i| {
                q.* = (if (i % 2 == 0) m.le(t) else m.lt(t)).select(level, q.*);
            }
            result.err = result.err.add(abs(q.*.sub(m)));
        }
        return result;
    }

    fn abs(x: Value) Value {
        return x.bitCast(.i32).bitAnd(0x7fffffff).bitCast(.f32);
    }
};

fn decodeE8m0(b: *Builder, byte: Value) Value {
    const scale = byte.toUnsigned(.i32);
    const finite = scale.eq(0).select(b.constant(.i32, 0x00400000), scale.shl(23));
    return scale.eq(255).select(b.constant(.i32, 0x7fc00000), finite).bitCast(.f32);
}

/// Two E4M3FN scales, low byte first. CDNA3 converts E4M3FNUZ, whose exponent
/// bias is one larger, so the same bits decode to half the value. Scales are
/// nonnegative and finite, avoiding the formats' differing NaN and -0 encodings.
fn decodeE4m3Pair(b: *Builder, pair: Value) Value {
    return b.cvtPkF32Fp8(pair.toUnsigned(.i32), false).mul(2.0);
}
