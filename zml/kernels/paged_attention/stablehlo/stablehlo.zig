const zml = @import("../../../zml.zig");
const KvCache = @import("../paged_attention.zig").KvCache;
const AttentionOptions = @import("../paged_attention.zig").AttentionOptions;
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
        .{
            .q = q.withPartitioning(opts.sharding.mesh, opts.sharding.partition(q.shape(), .hkv)),
            .kv_cache = .{ .split = .{
                .k = kv_cache.split.k.withPartitioning(opts.sharding.mesh, opts.sharding.kvPartition(kv_cache.split.k.shape(), .hkv)),
                .v = kv_cache.split.v.withPartitioning(opts.sharding.mesh, opts.sharding.kvPartition(kv_cache.split.v.shape(), .hkv)),
            } },
            .parameters = parameters,
            .opts = opts,
        },
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
                    const next_k_offset = state.k_page_idx.addConstant(1).scale(page_size);

                    // The current K page is processed in this iteration. Continue if the next page contains
                    // any key visible to the current query chunk, whose exclusive end is q_offset + num_slots.
                    const query_has_more_keys: zml.Tensor = .cmp(next_k_offset, .LT, q_offset.addConstant(num_slots));
                    const sequence_has_more_queries: zml.Tensor = .cmp(state.q_page_idx.addConstant(1).scale(num_slots), .LT, if_ctx.seq_num_queries);

                    const partial_softmax = while_body.attentionOnePage(state, q_offset, num_slots, page_size);
                    return updateState(state, num_slots, query_has_more_keys, sequence_has_more_queries, partial_softmax);
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
                    return updateState(state, num_slots, query_has_more_keys, sequence_has_more_queries, partial_softmax);
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
        const active_q = self.q.slice(self.q.axis(.b), .dyn(state.slot_id, q_chunk));

        const active_k_page_id = self.parameters.block_table.slices(
            .{ .b, .p },
            &.{ .dynSingle(state.seq_id), .dynSingle(state.k_page_idx) },
        );
        const active_k, const active_v = self.kv_cache.getPage(active_k_page_id);

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

    pub fn updateState(state: State, num_slots: u32, query_has_more_keys: zml.Tensor, sequence_has_more_queries: zml.Tensor, softmax: PartialSoftmax) State {
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
                .out = state.out.dynamicUpdateSlice(.{ .b = state.slot_id }, softmax.finalize().convert(state.out.dtype())),
                .slot_id = state.slot_id.addConstant(num_slots),
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
