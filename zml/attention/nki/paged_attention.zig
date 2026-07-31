const std = @import("std");
const platforms = @import("platforms");

const zml = @import("../../zml.zig");
const stdx = zml.stdx;

const paged_attention_source_path = "zml/zml/attention/nki/paged_attention.py";

pub const Options = struct {
    batch_size: usize,
    max_num_pages: usize,
    is_prefill: bool,
    compiler_target: []const u8,

    pub const InitOptions = struct {
        batch_size: usize,
        max_num_pages: usize,
        is_prefill: bool,
    };

    pub fn init(init_options: InitOptions) Options {
        if (comptime platforms.isEnabled(.neuron)) {
            const nki_kernel = @import("platforms/neuron/nki_kernel");
            const compiler_target = nki_kernel.compilerTarget();
            switch (compiler_target) {
                .trn2, .trn3 => {},
                .trn1, .trn1n, .inf2, .trn2n => stdx.debug.panic(
                    "Neuron paged attention requires a Trn2 or Trn3 instance, got {s}",
                    .{@tagName(compiler_target)},
                ),
            }
            return .{
                .batch_size = init_options.batch_size,
                .max_num_pages = init_options.max_num_pages,
                .is_prefill = init_options.is_prefill,
                .compiler_target = @tagName(compiler_target),
            };
        }

        stdx.debug.panic("NKI paged attention requires the Neuron platform", .{});
    }

    pub fn isPrefill(self: Options) bool {
        return self.is_prefill;
    }

    pub fn maxNumPages(self: Options) usize {
        return self.max_num_pages;
    }
};

pub const Parameters = struct {
    block_table: zml.Tensor,
    seq_lens: zml.Tensor,
    query_start_len: zml.Tensor,
    options_: Options,

    pub fn init(options_: Options) Parameters {
        return .{
            .block_table = .init(.{ .b = options_.batch_size, .p = options_.max_num_pages }, .i32),
            .seq_lens = .init(.{ .b = options_.batch_size }, .i32),
            .query_start_len = .init(.{ .b = options_.batch_size + 1 }, .i32),
            .options_ = options_,
        };
    }

    pub fn allocationSize(self: Parameters) usize {
        return self.block_table.byteSize() + self.seq_lens.byteSize() + self.query_start_len.byteSize();
    }

    pub fn options(self: Parameters) Options {
        return self.options_;
    }
};

pub const DecodeResult = struct {
    output: zml.Tensor,
    k_cache: zml.Tensor,
    v_cache: zml.Tensor,
};

pub const CacheUpdate = struct {
    slot_mapping: zml.Tensor,
    k: zml.Tensor,
    v: zml.Tensor,
};

/// Make padded cache rows safe for the Neuron scatter that precedes segmented
/// prefill. Invalid rows repeat the first real update and never address a sink
/// page or an out-of-range slot.
pub fn prepareCacheUpdate(slot_mapping: zml.Tensor, new_k: zml.Tensor, new_v: zml.Tensor, slot_count: usize) CacheUpdate {
    stdx.debug.assert(slot_count <= std.math.maxInt(i32), "Neuron KV cache has too many addressable slots: {}", .{slot_count});

    const zero = zml.Tensor.scalar(@as(i32, 0), .i32).broad(slot_mapping.shape());
    const limit = zml.Tensor.scalar(@as(i32, @intCast(slot_count)), .i32).broad(slot_mapping.shape());
    const valid = slot_mapping.cmp(.GE, zero)
        .logical(.AND, slot_mapping.cmp(.LT, limit));
    const first_slot = slot_mapping.slice1d(0, .single(0)).broad(slot_mapping.shape());
    const first_k = new_k.slice1d(0, .single(0)).broad(new_k.shape());
    const first_v = new_v.slice1d(0, .single(0)).broad(new_v.shape());

    return .{
        .slot_mapping = valid.select(slot_mapping, first_slot),
        .k = valid.broad(new_k.shape()).select(new_k, first_k),
        .v = valid.broad(new_v.shape()).select(new_v, first_v),
    };
}

/// Run the complete upstream TKG decode path. This is the platform contract
/// above the raw custom call: it owns projection packing, RoPE construction,
/// model sharding, and logical-to-physical multi-KV index conversion.
pub fn fusedDecode(
    parameters: Parameters,
    x: zml.Tensor,
    q_weight: zml.Tensor,
    k_weight: zml.Tensor,
    v_weight: zml.Tensor,
    k_cache: zml.Tensor,
    v_cache: zml.Tensor,
    token_pos: zml.Tensor,
    slot_mapping: zml.Tensor,
    o_weight: zml.Tensor,
    rope_opts: zml.nn.RopeOpts,
    num_query_heads: i64,
    num_kv_heads: i64,
) DecodeResult {
    const options = parameters.options_;
    stdx.debug.assert(!options.isPrefill(), "NKI fused decode requires decode parameters", .{});
    stdx.debug.assert(x.rank() == 2, "NKI fused decode expects X [tokens, hidden], got {f}", .{x});
    stdx.debug.assert(x.dtype() == .bf16, "NKI fused decode requires BF16 activations, got {s}", .{@tagName(x.dtype())});
    inline for (.{
        .{ "Q", q_weight },
        .{ "K", k_weight },
        .{ "V", v_weight },
        .{ "output", o_weight },
    }) |projection| {
        stdx.debug.assert(projection[1].rank() == 2, "NKI fused decode requires a dense {s} projection, got {f}", .{ projection[0], projection[1] });
        stdx.debug.assert(projection[1].dtype() == .bf16, "NKI fused decode requires a BF16 {s} projection, got {s}", .{ projection[0], @tagName(projection[1].dtype()) });
    }
    stdx.debug.assert(k_cache.rank() == 4 and v_cache.rank() == 4, "NKI fused decode expects caches [page, kv_head, chunk, head_dim], got K={f} V={f}", .{ k_cache, v_cache });
    stdx.debug.assert(k_cache.shape().eql(v_cache.shape()), "NKI fused decode requires matching K/V cache shapes, got K={f} V={f}", .{ k_cache, v_cache });
    stdx.debug.assert(k_cache.dtype() == .bf16, "NKI fused decode requires BF16 caches, got {s}", .{@tagName(k_cache.dtype())});
    stdx.debug.assert(k_cache.dim(.hkv) == num_kv_heads, "NKI fused decode cache has {} KV heads, expected {}", .{ k_cache.dim(.hkv), num_kv_heads });
    stdx.debug.assert(@mod(num_query_heads, num_kv_heads) == 0, "NKI fused decode requires query heads ({}) divisible by KV heads ({})", .{ num_query_heads, num_kv_heads });
    stdx.debug.assert(rope_opts.layout == .real_im_pass, "NKI fused decode requires contiguous half-split RoPE", .{});

    const compilation_context = zml.module.CompilationContext.current();
    const model_partitions = compilation_context.partitioning.numPartitionsForLogicalAxis(k_cache.shape(), .model) catch
        stdx.debug.panic("NKI fused decode requires the KV-head axis to use model partitioning", .{});
    stdx.debug.assert(@mod(num_kv_heads, model_partitions) == 0, "NKI fused decode requires KV heads ({}) divisible by model partitions ({})", .{ num_kv_heads, model_partitions });

    const batch_size: i64 = @intCast(options.batch_size);
    const token_count = x.dim(.b);
    stdx.debug.assert(@mod(token_count, batch_size) == 0, "decode token count {} is not divisible by batch size {}", .{ token_count, batch_size });
    const tokens_per_sequence = @divExact(token_count, batch_size);

    const x_bsh = x.splitAxis(.b, .{ .b = batch_size, .s = tokens_per_sequence })
        .withPartitioning(.{ .b = .replicated, .s = .replicated, .d = .replicated });
    const q_weight_sharded = q_weight.withPartitioning(.{ .dout = .model, .d = .replicated });
    const k_weight_sharded = k_weight.withPartitioning(.{ .dout = .model, .d = .replicated });
    const v_weight_sharded = v_weight.withPartitioning(.{ .dout = .model, .d = .replicated });
    const k_cache_sharded = k_cache.withPartitioning(.{ .page = .replicated, .k_chunk = .replicated, .hkv = .model, .hd = .replicated });
    const v_cache_sharded = v_cache.withPartitioning(.{ .page = .replicated, .k_chunk = .replicated, .hkv = .model, .hd = .replicated });
    const block_table_replicated = parameters.block_table.withPartitioning(.{ .b = .replicated, .p = .replicated });
    const o_weight_sharded = o_weight.withPartitioning(.{ .dout = .replicated, .d = .model });

    const token_pos_f32 = token_pos.convert(.f32).withTags(.{.token});
    const inv_freq = zml.nn.invFreq(k_cache.dim(.hd), rope_opts).withTags(.{.hd});
    const angles = zml.Tensor.outer(token_pos_f32, inv_freq);
    const cos = angles.cos().convert(x.dtype())
        .splitAxis(.token, .{ .b = batch_size, .s = tokens_per_sequence })
        .transpose(.{ .hd, .b, .s })
        .withPartitioning(.{ .hd = .replicated, .b = .replicated, .s = .replicated });
    const sin = angles.sin().convert(x.dtype())
        .splitAxis(.token, .{ .b = batch_size, .s = tokens_per_sequence })
        .transpose(.{ .hd, .b, .s })
        .withPartitioning(.{ .hd = .replicated, .b = .replicated, .s = .replicated });
    const pos_ids = token_pos.convert(.f32)
        .splitAxis(.b, .{ .b = batch_size, .s = tokens_per_sequence })
        .withPartitioning(.{ .b = .replicated, .s = .replicated });
    const cache_update_idx = slot_mapping
        .splitAxis(.b, .{ .b = batch_size, .s = tokens_per_sequence })
        .withPartitioning(.{ .b = .replicated, .s = .replicated });
    const output_shape = zml.Shape.init(.{ .b = token_count, .d = x.dim(.d) }, x.dtype())
        .withPartitioning(.{ .b = .replicated, .d = .replicated });
    const output_shapes: []const zml.Shape = &.{ output_shape, k_cache_sharded.shape(), v_cache_sharded.shape() };
    const outputs = zml.ops.manualComputation(
        .{
            x_bsh,
            q_weight_sharded,
            k_weight_sharded,
            v_weight_sharded,
            k_cache_sharded,
            v_cache_sharded,
            block_table_replicated,
            pos_ids,
            cache_update_idx,
            o_weight_sharded,
            cos,
            sin,
        },
        output_shapes,
        .{ .options = options },
        (struct {
            fn body(context: anytype, allocator: std.mem.Allocator, local_inputs: []const zml.Tensor, _: []const zml.Shape) []const zml.Tensor {
                stdx.debug.assert(local_inputs.len == 12, "NKI fused decode expects 12 local inputs, got {}", .{local_inputs.len});
                const local_x = local_inputs[0];
                const local_q_weight = local_inputs[1];
                const local_k_weight = local_inputs[2];
                const local_v_weight = local_inputs[3];
                const local_k_cache = local_inputs[4];
                const local_v_cache = local_inputs[5];

                const local_num_kv_heads = local_k_cache.dim(.hkv);
                stdx.debug.assert(local_num_kv_heads >= 1 and local_v_cache.dim(.hkv) == local_num_kv_heads, "NKI fused decode requires matching local KV heads, got K={} V={}", .{ local_num_kv_heads, local_v_cache.dim(.hkv) });
                const head_dim = local_k_cache.dim(.hd);
                const local_kv_projection_dim = local_num_kv_heads * head_dim;
                stdx.debug.assert(local_k_weight.dim(.dout) == local_kv_projection_dim and local_v_weight.dim(.dout) == local_kv_projection_dim, "NKI fused decode requires {} local K/V heads of dimension {}, got K={} V={}", .{ local_num_kv_heads, head_dim, local_k_weight.dim(.dout), local_v_weight.dim(.dout) });
                stdx.debug.assert(@mod(local_q_weight.dim(.dout), head_dim) == 0, "local Q projection dimension {} is not divisible by head dimension {}", .{ local_q_weight.dim(.dout), head_dim });
                const local_num_query_heads = @divExact(local_q_weight.dim(.dout), head_dim);
                stdx.debug.assert(@mod(local_num_query_heads, local_num_kv_heads) == 0, "local query heads ({}) must be divisible by local KV heads ({})", .{ local_num_query_heads, local_num_kv_heads });

                const local_w_qkv = zml.Tensor.concatenate(&.{ local_q_weight, local_k_weight, local_v_weight }, .dout)
                    .transpose(.{ .d, .dout });
                const local_w_out = local_inputs[9].transpose(.{ .d, .dout });
                const active_token_mask = zml.Tensor.constant(.{ .f32 = 1 }).broad(zml.Shape.init(.{
                    .q = local_x.dim(.s),
                    .b = local_x.dim(.b),
                    .h = local_num_query_heads,
                    .k = local_x.dim(.s),
                }, .f32));

                const result = decodeKernel(
                    context.options,
                    local_x,
                    local_w_qkv,
                    local_k_cache,
                    local_v_cache,
                    local_inputs[6],
                    active_token_mask,
                    local_inputs[7],
                    local_inputs[8],
                    local_w_out,
                    local_inputs[10],
                    local_inputs[11],
                );

                const local_outputs = allocator.alloc(zml.Tensor, 3) catch unreachable;
                local_outputs[0] = zml.ops.allReduce(result.output, zml.Tensor.add);
                local_outputs[1] = result.k_cache;
                local_outputs[2] = result.v_cache;
                return local_outputs;
            }
        }).body,
    );

    return .{ .output = outputs[0], .k_cache = outputs[1], .v_cache = outputs[2] };
}

/// Adapt grouped-query packed prefill tensors to the segmented NKI kernel.
pub fn pagedAttention(parameters: Parameters, q: zml.Tensor, k_cache: zml.Tensor, v_cache: zml.Tensor) zml.Tensor {
    const options = parameters.options_;
    stdx.debug.assert(options.isPrefill(), "NKI segmented attention requires prefill parameters", .{});
    stdx.debug.assert(q.rank() == 4, "NKI segmented prefill expects Q [tokens, kv_head, head_group, head_dim], got {f}", .{q});
    stdx.debug.assert(k_cache.rank() == 4 and v_cache.rank() == 4, "NKI segmented prefill expects caches [page, kv_head, chunk, head_dim], got K={f} V={f}", .{ k_cache, v_cache });
    stdx.debug.assert(k_cache.shape().eql(v_cache.shape()), "NKI segmented prefill requires matching K/V cache shapes, got K={f} V={f}", .{ k_cache, v_cache });

    const compilation_context = zml.module.CompilationContext.current();
    const model_partitions = compilation_context.partitioning.numPartitionsForLogicalAxis(k_cache.shape(), .model) catch
        stdx.debug.panic("NKI segmented prefill requires the KV-head axis to use model partitioning", .{});
    stdx.debug.assert(@mod(q.dim(.hkv), model_partitions) == 0, "NKI segmented prefill requires KV heads ({}) divisible by model partitions ({})", .{ q.dim(.hkv), model_partitions });

    const q_sharded = q.withPartitioning(.{ .b = .replicated, .hkv = .model, .hg = .replicated, .hd = .replicated });
    const k_cache_sharded = k_cache.withPartitioning(.{ .page = .replicated, .k_chunk = .replicated, .hkv = .model, .hd = .replicated });
    const v_cache_sharded = v_cache.withPartitioning(.{ .page = .replicated, .k_chunk = .replicated, .hkv = .model, .hd = .replicated });

    // The segmented kernel handles one sequence. The scheduler reserves the
    // entire shared batch for prefill, leaving row zero as the active row.
    const active_block_table = parameters.block_table.slice1d(.b, .{ .end = 1 })
        .withPartitioning(.{ .b = .replicated, .p = .replicated });
    const total_tokens = parameters.seq_lens.slice1d(.b, .{ .end = 1 });
    const query_start = parameters.query_start_len.slice1d(.b, .{ .end = 1 });
    const query_end = parameters.query_start_len.slice1d(.b, .{ .start = 1, .end = 2 });
    const prior_tokens = total_tokens.sub(query_end.sub(query_start))
        .reshape(zml.Shape.init(.{ .one = 1, .scalar = 1 }, .i32))
        .withPartitioning(.{ .one = .replicated, .scalar = .replicated });

    return zml.ops.manualComputation(
        .{ q_sharded, k_cache_sharded, v_cache_sharded, active_block_table, prior_tokens },
        q_sharded.shape(),
        .{ .options = options },
        (struct {
            fn body(context: anytype, _: std.mem.Allocator, local_inputs: []const zml.Tensor, _: []const zml.Shape) zml.Tensor {
                stdx.debug.assert(local_inputs.len == 5, "NKI segmented prefill expects 5 local inputs, got {}", .{local_inputs.len});
                const local_q = local_inputs[0];
                const local_k_cache = local_inputs[1];
                const local_v_cache = local_inputs[2];
                const local_num_kv_heads = local_q.dim(.hkv);
                stdx.debug.assert(local_num_kv_heads >= 1 and local_k_cache.dim(.hkv) == local_num_kv_heads and local_v_cache.dim(.hkv) == local_num_kv_heads, "NKI segmented prefill requires matching local KV heads, got Q={} K={} V={}", .{ local_num_kv_heads, local_k_cache.dim(.hkv), local_v_cache.dim(.hkv) });

                const local_num_head_groups = local_q.dim(.hg);
                const q_nki = local_q.transpose(.{ .hkv, .hg, .b, .hd })
                    .merge(.{ .h = .{ .hkv, .hg } });
                const output = prefillKernel(context.options, q_nki, local_k_cache, local_v_cache, local_inputs[3], local_inputs[4]);
                return output.splitAxis(.h, .{ .hkv = local_num_kv_heads, .hg = local_num_head_groups })
                    .transpose(local_q.shape());
            }
        }).body,
    );
}

/// Bind vLLM Neuron's fused decode megakernel.
///
/// The caller supplies a head-inner cache plus llmd's logical block and slot
/// indices. This adapter folds the local KV head into the physical indices
/// expected by the upstream multi-KV kernel.
fn decodeKernel(
    options: Options,
    x: zml.Tensor,
    w_qkv: zml.Tensor,
    k_cache: zml.Tensor,
    v_cache: zml.Tensor,
    logical_block_table: zml.Tensor,
    active_mask: zml.Tensor,
    pos_ids: zml.Tensor,
    logical_cache_slots: zml.Tensor,
    w_out: zml.Tensor,
    cos: zml.Tensor,
    sin: zml.Tensor,
) DecodeResult {
    stdx.debug.assert(!options.is_prefill, "paged_attention.decode requires decode options", .{});
    stdx.debug.assert(x.rank() == 3, "paged_attention.decode expects X [batch, tokens, hidden], got {f}", .{x});
    stdx.debug.assert(k_cache.rank() == 4 and v_cache.rank() == 4, "paged_attention.decode expects caches [page, kv_head, block, head_dim], got K={f} V={f}", .{ k_cache, v_cache });
    stdx.debug.assert(k_cache.shape().eql(v_cache.shape()), "paged_attention.decode expects matching K/V caches, got K={f} V={f}", .{ k_cache, v_cache });
    stdx.debug.assert(logical_block_table.rank() == 2, "paged_attention.decode expects logical block table [batch, page], got {f}", .{logical_block_table});
    stdx.debug.assert(logical_cache_slots.rank() == 2, "paged_attention.decode expects logical cache slots [batch, token], got {f}", .{logical_cache_slots});

    const kv_heads = k_cache.dim(.hkv);
    const block_size = k_cache.dim(.k_chunk);
    const block_table_shape = zml.Shape.init(.{
        .b = logical_block_table.dim(.b),
        .hkv = kv_heads,
        .p = logical_block_table.dim(.p),
    }, .i32);
    const logical_blocks = logical_block_table.insertAxes(.p, .{.hkv}).broad(block_table_shape);
    const physical_blocks = logical_blocks.scale(kv_heads).add(zml.Tensor.iota(block_table_shape, .hkv));
    const active_blocks_table = if (kv_heads == 1)
        logical_block_table
    else
        logical_blocks.cmp(.GE, zml.Tensor.scalar(@as(i32, 0), .i32).broad(block_table_shape)).select(
            physical_blocks,
            zml.Tensor.scalar(@as(i32, -1), .i32).broad(block_table_shape),
        );

    const zero_slots = zml.Tensor.scalar(@as(i32, 0), .i32).broad(logical_cache_slots.shape());
    const slot_limit = k_cache.dim(.page) * block_size;
    const valid_slots = logical_cache_slots.cmp(.GE, zero_slots).logical(
        .AND,
        logical_cache_slots.cmp(.LT, zml.Tensor.scalar(slot_limit, .i32).broad(logical_cache_slots.shape())),
    );
    const safe_slots = valid_slots.select(logical_cache_slots, zero_slots);
    const cache_update_idx = if (kv_heads == 1)
        valid_slots.select(
            safe_slots.convert(.u32),
            zml.Tensor.scalar(std.math.maxInt(u32), .u32).broad(safe_slots.shape()),
        )
    else b: {
        const update_shape = zml.Shape.init(.{
            .b = x.dim(.b),
            .hkv = kv_heads,
            .s = x.dim(.s),
        }, .i32);
        const slots = safe_slots.insertAxes(.s, .{.hkv}).broad(update_shape);
        const physical_slots = slots.divByConst(block_size)
            .scale(kv_heads * block_size)
            .add(zml.Tensor.iota(update_shape, .hkv).scale(block_size))
            .add(slots.remainder(zml.Tensor.scalar(block_size, .i32).broad(update_shape)))
            .convert(.u32);
        break :b valid_slots.insertAxes(.s, .{.hkv}).broad(update_shape).select(
            physical_slots,
            zml.Tensor.scalar(std.math.maxInt(u32), .u32).broad(physical_slots.shape()),
        );
    };

    const output_shape = zml.Shape.init(
        .{ .b = x.dim(0) * x.dim(1), .d = x.dim(2) },
        x.dtype(),
    );
    const outputs = zml.ops.neuronNki(
        .{ x, w_qkv, k_cache, v_cache, active_blocks_table, active_mask, pos_ids, cache_update_idx, w_out, cos, sin },
        .{ output_shape, k_cache.shape(), v_cache.shape() },
        .{
            .name = "paged_attention_decode",
            .entrypoint = "paged_attention_decode",
            .source_path = paged_attention_source_path,
            .compiler_target = options.compiler_target,
            .lnc = 2,
            .output_operand_aliases = &.{
                .{ .output_index = 1, .operand_index = 2 },
                .{ .output_index = 2, .operand_index = 3 },
            },
        },
    );
    return .{ .output = outputs[0], .k_cache = outputs[1], .v_cache = outputs[2] };
}

/// Bind vLLM Neuron's segmented paged-attention prefill kernel.
fn prefillKernel(
    options: Options,
    q: zml.Tensor,
    k_cache: zml.Tensor,
    v_cache: zml.Tensor,
    block_table: zml.Tensor,
    prior_tokens: zml.Tensor,
) zml.Tensor {
    stdx.debug.assert(options.is_prefill, "paged_attention.prefill requires prefill options", .{});
    stdx.debug.assert(q.rank() == 3, "paged_attention.prefill expects Q [batch_heads, tokens, head_dim], got {f}", .{q});
    stdx.debug.assert(k_cache.rank() == 4 and v_cache.rank() == 4, "paged_attention.prefill expects HND caches, got K={f} V={f}", .{ k_cache, v_cache });
    stdx.debug.assert(prior_tokens.rank() == 2 and prior_tokens.dim(0) == 1 and prior_tokens.dim(1) == 1, "paged_attention.prefill expects scalar prior_tokens [1, 1], got {f}", .{prior_tokens});

    // Neuron 2.31's segmented prefix-cache path requires scale=1.0 because it
    // uses range-select masking. Preserve scaled dot-product attention by
    // pre-scaling Q before the custom call.
    const attention_scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(q.dim(2)))));
    const scaled_q = q.scale(attention_scale);

    return zml.ops.neuronNki(
        .{ scaled_q, k_cache, v_cache, block_table, prior_tokens },
        .{q.shape()},
        .{
            .name = "paged_attention_prefill",
            .entrypoint = "paged_attention_prefill",
            .source_path = paged_attention_source_path,
            .compiler_target = options.compiler_target,
            .lnc = 2,
        },
    )[0];
}
