const std = @import("std");

const platforms = @import("platforms");

const zml = @import("../../zml.zig");
const stdx = zml.stdx;
const AttentionOptions = @import("../paged_attention.zig").AttentionOptions;

const paged_attention_source_path = "zml/zml/attention/nki/paged_attention.py";
const nki_tile_size_pmax = 128;

pub const Options = struct {
    batch_size: usize,
    max_num_pages: usize,
    max_token_count: usize,
    max_seqlen_q: usize,
    is_prefill: bool,
    compiler_target: []const u8,

    pub const InitOptions = struct {
        batch_size: usize,
        max_num_pages: usize,
        max_token_count: usize,
        max_seqlen_q: usize,
        is_prefill: bool,
    };

    pub fn init(init_options: InitOptions) Options {
        if (comptime platforms.isEnabled(.neuron)) {
            const nki_kernel = @import("platforms/neuron/nki_kernel");
            return .{
                .batch_size = init_options.batch_size,
                .max_num_pages = init_options.max_num_pages,
                .max_token_count = init_options.max_token_count,
                .max_seqlen_q = init_options.max_seqlen_q,
                .is_prefill = init_options.is_prefill,
                .compiler_target = @tagName(nki_kernel.compilerTargetFromInstance()),
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

pub const Context = struct {
    is_prefill: bool,
    max_seqlen_q: usize,

    pub fn init(parameters: Parameters, num_heads: i64, num_kv_heads: i64, head_dim: i64, page_size: i64) Context {
        _ = num_heads;
        _ = num_kv_heads;
        _ = head_dim;
        _ = page_size;
        return .{ .is_prefill = parameters.options_.is_prefill, .max_seqlen_q = parameters.options_.max_seqlen_q };
    }
};

pub fn pagedAttention(parameters: Parameters, context: Context, q: zml.Tensor, k_cache: zml.Tensor, v_cache: zml.Tensor, new_k: zml.Tensor, new_v: zml.Tensor, slot_mapping: ?zml.Tensor, opts: AttentionOptions) zml.Tensor {
    _ = context;
    _ = opts;

    if (parameters.options_.is_prefill) {
        const heads_per_kv: usize = @intCast(q.dim(.hg));
        const q_per_head_tile = nki_tile_size_pmax / heads_per_kv;
        stdx.debug.assert(@mod(nki_tile_size_pmax, heads_per_kv) == 0 and @mod(@as(usize, @intCast(q.dim(.b))), q_per_head_tile) == 0, "NKI prefill paged attention needs grouped Q rows to fill a 128-lane tile, got {f}", .{q});
    } else {
        stdx.debug.assert(slot_mapping != null, "NKI decode paged attention expects slot_mapping", .{});
    }

    const q_sharded = q.withPartitioning(.{ .b = .replicated, .hkv = .model, .hg = .replicated, .hd = .replicated });
    const k_cache_sharded = k_cache.withPartitioning(.{ .page = .replicated, .k_chunk = .replicated, .hkv = .model, .hd = .replicated });
    const v_cache_sharded = v_cache.withPartitioning(.{ .page = .replicated, .k_chunk = .replicated, .hkv = .model, .hd = .replicated });
    const new_k_sharded = new_k.withPartitioning(.{ .b = .replicated, .hkv = .model, .hd = .replicated });
    const new_v_sharded = new_v.withPartitioning(.{ .b = .replicated, .hkv = .model, .hd = .replicated });

    const seq_lens = parameters.seq_lens.insertAxes(.last, .{.one});
    const query_start_len = parameters.query_start_len.insertAxes(.last, .{.one});

    return zml.ops.manualComputation(
        .{
            q_sharded,
            k_cache_sharded,
            v_cache_sharded,
            new_k_sharded,
            new_v_sharded,
            (slot_mapping orelse parameters.query_start_len).withPartitioning(.{ .b = .replicated }),
            parameters.block_table.withPartitioning(.{ .b = .replicated, .p = .replicated }),
            seq_lens.withPartitioning(.{ .b = .replicated }),
            query_start_len.withPartitioning(.{ .b = .replicated }),
        },
        q_sharded.shape(),
        parameters,
        (struct {
            fn body(parameters_: Parameters, _: std.mem.Allocator, sharded_inputs: []const zml.Tensor, _: zml.Shape) zml.Tensor {
                return zml.ops.neuronNki(
                    .{ sharded_inputs[0], sharded_inputs[1], sharded_inputs[2], sharded_inputs[3], sharded_inputs[4], sharded_inputs[5], sharded_inputs[6], sharded_inputs[7], sharded_inputs[8] },
                    .{sharded_inputs[0].shape()},
                    .{
                        .name = if (parameters_.options_.is_prefill) "paged_attention_2d_trn1" else "paged_attention_decode_2d_trn1",
                        .entrypoint = if (parameters_.options_.is_prefill) "paged_attention_2d" else "paged_attention_decode_2d",
                        .source_path = paged_attention_source_path,
                        .compiler_target = parameters_.options_.compiler_target,
                        .has_side_effect = true,
                    },
                )[0];
            }
        }).body,
    );
}

pub const KvCacheUpdate = struct {
    k: zml.Tensor,
    v: zml.Tensor,
};

pub fn updateKvCachePaged(parameters: Parameters, k_cache: zml.Tensor, v_cache: zml.Tensor, new_k: zml.Tensor, new_v: zml.Tensor, slot_mapping: zml.Tensor) KvCacheUpdate {
    if (!parameters.options_.is_prefill) {
        return .{ .k = k_cache, .v = v_cache };
    }

    const k_cache_sharded = k_cache.withPartitioning(.{ .page = .replicated, .k_chunk = .replicated, .hkv = .model, .hd = .replicated });
    const v_cache_sharded = v_cache.withPartitioning(.{ .page = .replicated, .k_chunk = .replicated, .hkv = .model, .hd = .replicated });
    const new_k_sharded = new_k.withPartitioning(.{ .b = .replicated, .hkv = .model, .hd = .replicated });
    const new_v_sharded = new_v.withPartitioning(.{ .b = .replicated, .hkv = .model, .hd = .replicated });

    const outputs = zml.ops.manualComputation(
        .{
            k_cache_sharded,
            v_cache_sharded,
            new_k_sharded,
            new_v_sharded,
            slot_mapping.withPartitioning(.{ .b = .replicated }),
            parameters.query_start_len.insertAxes(.last, .{.one}).withPartitioning(.{ .b = .replicated, .one = .replicated }),
        },
        .{ k_cache_sharded.shape(), v_cache_sharded.shape() },
        parameters,
        (struct {
            fn body(parameters_: Parameters, allocator: std.mem.Allocator, sharded_inputs: []const zml.Tensor, output: []const zml.Shape) []const zml.Tensor {
                const outputs = zml.ops.neuronNki(
                    .{ sharded_inputs[0], sharded_inputs[1], sharded_inputs[2], sharded_inputs[3], sharded_inputs[4], sharded_inputs[5] },
                    .{ output[0], output[1] },
                    .{
                        .name = "paged_kv_cache_update_trn1",
                        .entrypoint = "paged_kv_cache_update",
                        .source_path = paged_attention_source_path,
                        .compiler_target = parameters_.options_.compiler_target,
                        .output_operand_aliases = &.{
                            .{ .output_index = 0, .operand_index = 0 },
                            .{ .output_index = 1, .operand_index = 1 },
                        },
                    },
                );
                const results = allocator.alloc(zml.Tensor, 2) catch unreachable;
                results[0] = outputs[0];
                results[1] = outputs[1];
                return results;
            }
        }).body,
    );

    return .{ .k = outputs[0], .v = outputs[1] };
}
