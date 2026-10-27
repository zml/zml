const std = @import("std");

const zml = @import("../../../zml.zig");
const KvCache = @import("../paged_attention.zig").KvCache;
const AttentionOptions = @import("../paged_attention.zig").AttentionOptions;
const Parameters = @import("../paged_attention.zig").Parameters;
const triton = @import("../triton/triton.zig");

pub fn pagedAttention(
    parameters: triton.paged.Parameters,
    q: zml.Tensor,
    kv_cache: KvCache,
    opts: AttentionOptions,
) zml.Tensor {
    switch (opts.mask) {
        .standard => |options| if (options.sliding_window != null) @panic("Sliding window attention mask is not supported on StableHLO"),
        .indices => @panic("Sparse attention mask is not supported on StableHLO"),
    }

    const page_size = kv_cache.split.k.dim(.k_chunk);

    const final_state = zml.ops.@"while"(
        AttentionLoop,
        .{ .q = q, .kv_cache = kv_cache, .parameters = parameters, .opts = opts },
        .{
            // i32 to match triton conventions
            .slot_id = .scalar(0, .i32),
            .seq_id = .scalar(0, .i32),
            .q_page_idx = .scalar(0, .i32),
            .k_page_idx = .scalar(0, .i32),
            // Note: we do all partial softmax in f32
            .partial_softmax_prefill = .zeroes(q.shape().setDim(.b, page_size).withDtype(.f32)),
            .partial_softmax_decode = .zeroes(q.shape().setDim(.b, 1).withDtype(.f32)),
            .out = .zeroes(q.shape()),
        },
    );

    return final_state.out;
}

const AttentionLoop = struct {
    q: zml.Tensor,
    kv_cache: KvCache,
    parameters: triton.paged.Parameters,
    opts: AttentionOptions,

    const While = @This();

    pub const State = struct {
        slot_id: zml.Tensor,
        seq_id: zml.Tensor,
        q_page_idx: zml.Tensor,
        k_page_idx: zml.Tensor,
        partial_softmax_prefill: PartialSoftmax,
        partial_softmax_decode: PartialSoftmax,
        out: zml.Tensor,
    };

    pub fn cond(self: While, state: State) zml.Tensor {
        const has_more_sequences = state.seq_id.cmp(.LT, .scalar(self.parameters.seq_lens.count(), .i32));
        const query_start = self.parameters.query_start_len.slice(0, .dynSingle(state.seq_id));
        const query_end = self.parameters.query_start_len.slice(0, .dynSingle(state.seq_id.addConstant(1)));
        const has_queries = query_start.cmp(.LT, query_end).asScalar();
        return has_more_sequences.logical(.AND, has_queries);
    }

    pub fn body(self: While, state_: State) State {
        const query_start = self.parameters.query_start_len.slice(0, .dynSingle(state_.seq_id));
        const query_end = self.parameters.query_start_len.slice(0, .dynSingle(state_.seq_id.addConstant(1)));
        const seq_num_queries_: zml.Tensor = .asScalar(.sub(query_end, query_start));
        const page_size_: u32 = @intCast(self.kv_cache.split.k.dim(.k_chunk));

        return zml.ops.@"if"(
            struct {
                const IfPrefill = @This();
                while_body: While,
                state: State,
                seq_num_queries: zml.Tensor,
                page_size: u32,

                /// Prefill version: num_slots == page_size, we multiply a full page of queries with a page of keys.
                pub fn onTrue(if_ctx: IfPrefill) State {
                    const while_body = if_ctx.while_body;
                    const state = if_ctx.state;
                    const page_size = if_ctx.page_size;
                    const num_slots: u32 = page_size;

                    // seq_lens includes the current queries; subtracting their count gives the context length.
                    const seq_len = while_body.parameters.seq_lens.slice(0, .dynSingle(state.seq_id));
                    const q_offset = seq_len.sub(if_ctx.seq_num_queries).add(state.q_page_idx.scale(page_size));
                    const valid_slots = if_ctx.seq_num_queries.sub(state.q_page_idx.scale(page_size)).minimum(.scalar(num_slots, .i32));
                    const next_k_offset = state.k_page_idx.addConstant(1).scale(page_size);

                    // The current K page is processed in this iteration. Continue if the next page contains
                    // any key visible to the valid queries in the current chunk.
                    const query_has_more_keys: zml.Tensor = .cmp(next_k_offset, .LT, q_offset.add(valid_slots));
                    const sequence_has_more_queries: zml.Tensor = .cmp(state.q_page_idx.addConstant(1).scale(num_slots), .LT, if_ctx.seq_num_queries);

                    const partial_softmax = while_body.attentionOnePage(state, q_offset, num_slots, page_size);
                    return updateState(state, valid_slots, num_slots, query_has_more_keys, sequence_has_more_queries, partial_softmax);
                }

                /// Decode version: num_slots == 1, we multiply one query with a page of keys.
                pub fn onFalse(if_ctx: IfPrefill) State {
                    const while_body = if_ctx.while_body;
                    const state = if_ctx.state;
                    const page_size = if_ctx.page_size;
                    const num_slots: u32 = 1;

                    // A decode query is the final token in seq_len, so its zero-based offset is seq_len - 1.
                    const seq_len = while_body.parameters.seq_lens.slice(0, .dynSingle(state.seq_id)).asScalar();
                    const q_offset = seq_len.sub(if_ctx.seq_num_queries);
                    const next_k_offset = state.k_page_idx.addConstant(1).scale(page_size);

                    const query_has_more_keys: zml.Tensor = .cmp(next_k_offset, .LT, q_offset.addConstant(num_slots));
                    const sequence_has_more_queries: zml.Tensor = .scalar(false, .bool);

                    const partial_softmax = while_body.attentionOnePage(state, q_offset, num_slots, page_size);
                    return updateState(state, .scalar(1, .i32), num_slots, query_has_more_keys, sequence_has_more_queries, partial_softmax);
                }
            },
            .{
                .while_body = self,
                .state = state_,
                .seq_num_queries = seq_num_queries_,
                .page_size = page_size_,
            },
            .cmp(seq_num_queries_, .GT, .scalar(1, .i32)),
        );
    }

    pub fn attentionOnePage(self: While, state: State, q_offset: zml.Tensor, q_chunk: u32, page_size: u32) PartialSoftmax {
        const k_offset = state.k_page_idx.scale(page_size);

        // Gather within this sequence, then zero the unused rows of its query page.
        const query_end = self.parameters.query_start_len.slice(0, .dynSingle(state.seq_id.addConstant(1))).asScalar();
        const q_indices = zml.Tensor.iota(.init(.{ .q_chunk = q_chunk }, .i32), .q_chunk).add(state.slot_id);
        const valid_q = q_indices.cmp(.LT, query_end);
        const active_q = self.q.gather(.{ .b = q_indices.minimum(query_end.addConstant(-1)) }, .{}).mask(valid_q, 0).rename(.{ .q_chunk = .b });

        const active_k_page_id = self.parameters.block_table.slices(
            .{ .b, .p },
            &.{ .dynSingle(state.seq_id), .dynSingle(state.k_page_idx) },
        );
        const k, const v = self.kv_cache.getPage(active_k_page_id);
        const valid_k = zml.Tensor.iota(.init(.{ .k_chunk = page_size }, .i32), .k_chunk).add(k_offset).cmp(.LT, self.parameters.seq_lens.slice(0, .dynSingle(state.seq_id)).asScalar());
        const active_k, const active_v = .{ k.mask(valid_k, 0), v.mask(valid_k, 0) };

        const dtype = self.q.dtype();
        const attn_mask: zml.Tensor = attn_mask: {
            const mask_shape: zml.Shape = .init(.{ .b = q_chunk, .k_chunk = page_size }, dtype);

            const q_idx = zml.Tensor.iota(mask_shape, .b).add(q_offset);
            const k_idx = zml.Tensor.iota(mask_shape, .k_chunk).add(k_offset);

            const mask = zml.Tensor.cmp(k_idx, .LE, q_idx);
            const minus_inf = zml.Tensor.constant(dtype.minValue());
            break :attn_mask .select(mask, .scalar(0, dtype), minus_inf);
        };

        const scale: f32 = self.opts.scale orelse @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(active_q.dim(.hd)))));
        var attn_weights = active_q.dot(active_k, .hd).scale(scale);
        attn_weights = attn_weights.add(attn_mask.broad(attn_weights.shape()));

        const og_softmax = if (q_chunk == 1) state.partial_softmax_decode else state.partial_softmax_prefill;
        // Note: we compute q * k using q and k dtype, but the partial softmax is in f32.
        const precise_dt = og_softmax.values.dtype();
        var partial = partialSoftmax(attn_weights.convert(precise_dt), .k_chunk);
        partial.values = partial.values.dot(active_v.convert(precise_dt), .k_chunk).transpose(og_softmax.values.shape());
        partial.max_value = partial.max_value.transpose(og_softmax.max_value.shape());
        partial.exp_sum = partial.exp_sum.transpose(og_softmax.exp_sum.shape());

        return og_softmax.merge(partial);
    }

    pub fn updateState(state: State, valid_slots: zml.Tensor, num_slots: u32, query_has_more_keys: zml.Tensor, sequence_has_more_queries: zml.Tensor, softmax: PartialSoftmax) State {
        const rows = zml.Tensor.iota(.init(.{ .q_chunk = num_slots }, .i32), .q_chunk);
        // Scatter drops unused rows instead of shifting a full-page write backward.
        const indices = rows.cmp(.LT, valid_slots).select(rows.add(state.slot_id), .scalar(std.math.maxInt(i32), .i32));
        return zml.ops.if2(
            query_has_more_keys,
            // If the query has more pages: update partial softmax with attention on current page, delay updating output.
            State{
                .slot_id = state.slot_id,
                .seq_id = state.seq_id,
                .q_page_idx = state.q_page_idx,
                .k_page_idx = state.k_page_idx.addConstant(1),
                .partial_softmax_prefill = if (num_slots == 1) state.partial_softmax_prefill else softmax,
                .partial_softmax_decode = if (num_slots == 1) softmax else state.partial_softmax_decode,
                .out = state.out,
            },
            // Query has no more pages, we commit softmax to output and increment slot
            State{
                .k_page_idx = .scalar(0, .i32),
                .partial_softmax_prefill = if (num_slots == 1) state.partial_softmax_prefill else .reset(softmax),
                .partial_softmax_decode = if (num_slots == 1) .reset(softmax) else state.partial_softmax_decode,
                .out = state.out.scatterSlices(
                    .{ .b = indices },
                    softmax.finalize().convert(state.out.dtype()).rename(.{ .b = .q_chunk }),
                    .{ .update_fn = zml.Tensor.ScatterOpts.override },
                ),
                .slot_id = state.slot_id.add(valid_slots),
                // If the sequence has more query chunks: increment q_page_idx, otherwise start over at 0.
                .q_page_idx = .select(sequence_has_more_queries, state.q_page_idx.addConstant(1), .scalar(0, .i32)),
                // If the sequence has more query chunks: keep seq_id, otherwise increment seq_id.
                .seq_id = .select(sequence_has_more_queries, state.seq_id, state.seq_id.addConstant(1)),
            },
        );
    }
};

pub const PartialSoftmax = struct {
    values: zml.Tensor,
    exp_sum: zml.Tensor,
    max_value: zml.Tensor,

    pub fn zeroes(q_shape: zml.Shape) PartialSoftmax {
        return .{
            .values = .zeroes(q_shape),
            .exp_sum = .zeroes(q_shape.drop(.hd)),
            .max_value = .broad(.constant(q_shape.dtype().minValue()), q_shape.drop(.hd)),
        };
    }

    pub fn reset(self: PartialSoftmax) PartialSoftmax {
        return .zeroes(self.values.shape());
    }

    pub fn merge(self: PartialSoftmax, other: PartialSoftmax) PartialSoftmax {
        // Rescale self and other using the new global_max.
        const global_max = self.max_value.maximum(other.max_value.convert(self.max_value.dtype()));
        const new_self = self.rescale(global_max);
        const new_other = other.rescale(global_max);

        // Now that self and other are using the same scale, we can just add them:
        return .{
            .max_value = global_max,
            .values = new_self.values.add(new_other.values),
            .exp_sum = new_self.exp_sum.add(new_other.exp_sum),
        };
    }

    /// Update max_value and rescale attn and exp_sum accordingly.
    pub fn rescale(self: PartialSoftmax, max_value: zml.Tensor) PartialSoftmax {
        const max_diff_exp: zml.Tensor = .exp(self.max_value.sub(max_value));
        const sum_dtype = self.exp_sum.dtype();
        return .{
            .max_value = max_value,
            .values = self.values.mul(max_diff_exp.broad(self.values.shape())),
            .exp_sum = self.exp_sum.mul(max_diff_exp.convert(sum_dtype)),
        };
    }

    /// Divides the intermediary results by the exp_sum to get the proper attention values.
    pub fn finalize(self: PartialSoftmax) zml.Tensor {
        return self.values.div(self.exp_sum.broad(self.values.shape()).convert(self.values.dtype()));
    }
};

/// Compute softmax over a chunk.
/// Returns intermediary results to allow aggregating later.
pub fn partialSoftmax(self: zml.Tensor, axis: anytype) PartialSoftmax {
    const a = self.axis(axis);
    const max_val = self.max(a).squeeze(a).maximum(.scalar(-1e16, self.dtype()));
    const out = self.sub(max_val.broad(self.shape())).exp();
    return .{
        .values = out,
        .exp_sum = out.convert(.f32).sum(a).squeeze(a),
        .max_value = max_val,
    };
}

const PagedAttentionTestData = struct {
    const q: [9][2]f32 = .{
        .{ 0.125, -0.25 },
        .{ 0.25, -0.5 },
        .{ 0.375, -0.75 },
        .{ 0.5, -0.25 },
        .{ 0.625, -0.5 },
        .{ 0.75, -0.75 },
        .{ 0.875, -0.25 },
        .{ 1, -0.5 },
        .{ 1.125, -0.75 },
    };
    const k: [8][2]f32 = .{
        .{ 0.125, 0 },
        .{ 0.25, 0.25 },
        .{ 0.375, 0.5 },
        .{ 0.5, 0 },
        .{ 0.625, 0.25 },
        .{ 0.75, 0.5 },
        .{ 0.875, 0 },
        .{ 1, 0.25 },
    };
    const v: [3][8][2]f32 = .{
        .{
            .{ 1, 1 },
            .{ 2, -1 },
            .{ 3, 1 },
            .{ 4, -1 },
            .{ 5, 1 },
            .{ 6, -1 },
            .{ 7, 1 },
            .{ 8, -1 },
        },
        .{
            .{ 11, 1 },
            .{ 12, -1 },
            .{ 13, 1 },
            .{ 14, -1 },
            .{ 15, 1 },
            .{ 16, -1 },
            .{ 17, 1 },
            .{ 18, -1 },
        },
        .{
            .{ 21, 1 },
            .{ 22, -1 },
            .{ 23, 1 },
            .{ 24, -1 },
            .{ 25, 1 },
            .{ 26, -1 },
            .{ 27, 1 },
            .{ 28, -1 },
        },
    };
};

test "StableHLO paged attention handles a partial final query page" {
    const case: TestCase = .{ .query_counts = .{ 6, 0, 0 }, .capacity = 6 };
    try case.run();
}

test "StableHLO paged attention advances between packed sequences" {
    const case: TestCase = .{
        .query_counts = .{ 6, 2, 1 },
        .context_lengths = .{ 0, 1, 2 },
        .capacity = 9,
    };
    try case.run();
}

test "StableHLO paged attention handles a partial page after a cached prefix" {
    const case: TestCase = .{
        .query_counts = .{ 2, 0, 0 },
        .context_lengths = .{ 1, 0, 0 },
        .capacity = 4,
        .check_single_page = true,
    };
    try case.run();
}

test "StableHLO paged attention ignores NaNs in unused keys and values during decode" {
    const case: TestCase = .{
        .query_counts = .{ 1, 1, 1 },
        .context_lengths = .{ 2, 4, 7 },
        .capacity = 4,
    };
    try case.run();
}

test "StableHLO paged attention zeroes unused output rows" {
    const case: TestCase = .{ .query_counts = .{ 6, 0, 0 }, .capacity = 10 };
    try case.run();
}

test "StableHLO paged attention handles complete query pages" {
    const case: TestCase = .{ .query_counts = .{ 4, 4, 0 }, .capacity = 8 };
    try case.run();
}

test "StableHLO paged attention handles a query buffer smaller than one page" {
    const case: TestCase = .{ .query_counts = .{ 2, 0, 0 }, .capacity = 2 };
    try case.run();
}

const TestCase = struct {
    query_counts: [3]u32,
    context_lengths: [3]u32 = .{ 0, 0, 0 },
    capacity: u32,
    check_single_page: bool = false,

    fn run(self: TestCase) !void {
        const platform = zml.testing.env();
        const allocator = std.testing.allocator;
        const io = std.testing.io;
        const page_size = 4;
        const parameters = Parameters.init(.fromBackend(.{
            .backend = .stablehlo,
            .is_prefill = true,
            .batch_size = 3,
            .seq_len = 12,
            .max_num_pages = 3,
            .max_token_count = self.capacity,
            .num_heads = 1,
            .num_kv_heads = 1,
            .head_dim = 2,
            .max_seqlen_q = self.capacity,
        })).stablehlo;
        const q: zml.Tensor = .init(.{ .b = self.capacity, .hkv = 1, .hg = 1, .hd = 2 }, .f32);
        const kv_shape: zml.Shape = .init(.{ .page = 10, .k_chunk = page_size, .hkv = 1, .hd = 2 }, .f32);
        const kv_cache: KvCache = .{ .split = .{ .k = .fromShape(kv_shape), .v = .fromShape(kv_shape) } };

        var q_data: [12][2]f32 = @splat(@splat(std.math.nan(f32)));
        var k_data: [10][page_size][2]f32 = @splat(@splat(@splat(std.math.nan(f32))));
        var v_data: [10][page_size][2]f32 = @splat(@splat(@splat(std.math.nan(f32))));
        var seq_lens: [3]i32 = undefined;
        var query_start_len: [4]i32 = .{ 0, 0, 0, 0 };
        var block_table: [3][3]i32 = @splat(@splat(-1));
        // Keep physical pages out of sequence order and leave page zero unused.
        const pages: [3][3]usize = .{ .{ 3, 1, 7 }, .{ 8, 2, 6 }, .{ 5, 9, 4 } };
        for (self.query_counts, self.context_lengths, 0..) |query_count, context_length, seq| {
            const seq_len = context_length + query_count;
            seq_lens[seq] = @intCast(seq_len);
            query_start_len[seq + 1] = query_start_len[seq] + @as(i32, @intCast(query_count));
            for (0..seq_len) |pos| {
                const page = pages[seq][pos / page_size];
                block_table[seq][pos / page_size] = @intCast(page);
                k_data[page][pos % page_size] = PagedAttentionTestData.k[pos];
                v_data[page][pos % page_size] = PagedAttentionTestData.v[seq][pos];
            }
        }
        const query_count: usize = @intCast(query_start_len[3]);
        try std.testing.expect(query_count <= self.capacity);
        @memcpy(q_data[0..query_count], PagedAttentionTestData.q[0..query_count]);
        const expected = referencePagedAttention(q_data[0..self.capacity], &k_data, &v_data, &block_table, &seq_lens, &query_start_len);

        var parameters_d: zml.Bufferized(triton.paged.Parameters) = .{
            .block_table = try .fromBytes(io, platform, parameters.block_table.shape(), std.mem.sliceAsBytes(&block_table)),
            .seq_lens = try .fromBytes(io, platform, parameters.seq_lens.shape(), std.mem.sliceAsBytes(&seq_lens)),
            .query_start_len = try .fromBytes(io, platform, parameters.query_start_len.shape(), std.mem.sliceAsBytes(&query_start_len)),
        };
        defer zml.Buffer.deinitAll(triton.paged.Parameters, &parameters_d);
        var q_d = try zml.Buffer.fromBytes(io, platform, q.shape(), std.mem.sliceAsBytes(q_data[0..self.capacity]));
        defer q_d.deinit();
        var k_d = try zml.Buffer.fromBytes(io, platform, kv_shape, std.mem.sliceAsBytes(&k_data));
        defer k_d.deinit();
        var v_d = try zml.Buffer.fromBytes(io, platform, kv_shape, std.mem.sliceAsBytes(&v_data));
        defer v_d.deinit();
        const kv_cache_d: zml.Bufferized(KvCache) = .{ .split = .{ .k = k_d, .v = v_d } };
        var exe = try platform.compileFn(allocator, io, pagedAttention, .{ parameters, q, kv_cache, .{ .scale = 1 } }, .{});
        defer exe.deinit();
        var output = try exe.eval(allocator, io, .{ parameters_d, q_d, kv_cache_d, .{} });
        defer output.deinit();
        try zml.testing.expectEqualShapes(q.shape(), output.shape());
        try zml.testing.expectClose(io, zml.Slice.init(q.shape(), std.mem.sliceAsBytes(expected[0..self.capacity])), output, .{
            .absolute_tolerance = 1e-5,
            .relative_tolerance = 1e-5,
            .minimum_close_fraction = 1,
        });

        // Zeroing unused K/V entries can hide an extra page visit in the output.
        // When all keys fit in one page, one step must finish the sequence.
        if (self.check_single_page) {
            const Step = struct {
                fn forward(params: triton.paged.Parameters, queries: zml.Tensor, cache: KvCache) zml.Tensor {
                    const loop: AttentionLoop = .{ .q = queries, .kv_cache = cache, .parameters = params, .opts = .{} };
                    return loop.body(.{
                        .slot_id = .scalar(0, .i32),
                        .seq_id = .scalar(0, .i32),
                        .q_page_idx = .scalar(0, .i32),
                        .k_page_idx = .scalar(0, .i32),
                        .partial_softmax_prefill = .zeroes(queries.shape().setDim(.b, cache.split.k.dim(.k_chunk)).withDtype(.f32)),
                        .partial_softmax_decode = .zeroes(queries.shape().setDim(.b, 1).withDtype(.f32)),
                        .out = .zeroes(queries.shape()),
                    }).seq_id;
                }
            };
            var step_exe = try platform.compileFn(allocator, io, Step.forward, .{ parameters, q, kv_cache }, .{});
            defer step_exe.deinit();
            var next_seq = try step_exe.eval(allocator, io, .{ parameters_d, q_d, kv_cache_d });
            defer next_seq.deinit();
            try std.testing.expectEqual(@as(i32, 1), try next_seq.getValue(i32, io));
        }
    }
};

// Scalar reference reads the actual paged buffers, independently of AttentionLoop.
fn referencePagedAttention(
    q: []const [2]f32,
    k: []const [4][2]f32,
    v: []const [4][2]f32,
    block_table: []const [3]i32,
    seq_lens: []const i32,
    query_start_len: []const i32,
) [12][2]f32 {
    var expected: [12][2]f32 = @splat(@splat(0));
    for (seq_lens, 0..) |seq_len, seq| {
        const start: usize = @intCast(query_start_len[seq]);
        const end: usize = @intCast(query_start_len[seq + 1]);
        const context_length = @as(usize, @intCast(seq_len)) - (end - start);
        for (start..end) |slot| {
            const key_count = context_length + slot - start + 1;
            var scores: [12]f64 = undefined;
            var max_score = -std.math.inf(f64);
            for (scores[0..key_count], 0..) |*score, pos| {
                const page: usize = @intCast(block_table[seq][pos / 4]);
                score.* = 0;
                for (q[slot], k[page][pos % 4]) |query, key| {
                    score.* += @as(f64, query) * @as(f64, key);
                }
                max_score = @max(max_score, score.*);
            }

            var weight_sum: f64 = 0;
            var values: [2]f64 = @splat(0);
            for (scores[0..key_count], 0..) |score, pos| {
                const page: usize = @intCast(block_table[seq][pos / 4]);
                const weight = @exp(score - max_score);
                weight_sum += weight;
                for (&values, v[page][pos % 4]) |*value, cached_value| {
                    value.* += weight * @as(f64, cached_value);
                }
            }
            for (&expected[slot], values) |*value, sum| {
                value.* = @floatCast(sum / weight_sum);
            }
        }
    }
    return expected;
}
