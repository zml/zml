const std = @import("std");

const platforms = @import("platforms");

const zml = @import("../../zml.zig");
const stdx = zml.stdx;
const AttentionOptions = @import("../paged_attention.zig").AttentionOptions;

const KernelSpec = struct {
    name: []const u8,
    entrypoint: []const u8,
    source_path: []const u8,
};

pub const Options = struct {
    batch_size: usize,
    max_num_pages: usize,
    max_seqlen_q: usize,
    is_prefill: bool,
    compiler_target: []const u8,
    attention_kernel: KernelSpec,
    kv_update_kernel: KernelSpec,

    pub const InitOptions = struct {
        batch_size: usize,
        max_num_pages: usize,
        max_seqlen_q: usize,
        is_prefill: bool,
    };

    pub fn init(init_options: InitOptions) Options {
        if (comptime platforms.isEnabled(.neuron)) {
            const nki_kernel = @import("platforms/neuron/nki_kernel");
            const compiler_target = nki_kernel.compilerTargetFromInstance();
            return .{
                .batch_size = init_options.batch_size,
                .max_num_pages = init_options.max_num_pages,
                .max_seqlen_q = init_options.max_seqlen_q,
                .is_prefill = init_options.is_prefill,
                .compiler_target = @tagName(compiler_target),
                .attention_kernel = switch (compiler_target) {
                    .trn1, .inf2, .trn1n, .trn2, .trn2n, .trn3 => .{
                        .name = "paged_attention_2d_trn1",
                        .entrypoint = "paged_attention_2d",
                        .source_path = "zml/zml/attention/nki/paged_attention.py",
                    },
                },
                .kv_update_kernel = switch (compiler_target) {
                    .trn1, .inf2, .trn1n, .trn2, .trn2n, .trn3 => .{
                        .name = "paged_kv_cache_update_trn1",
                        .entrypoint = "paged_kv_cache_update",
                        .source_path = "zml/zml/attention/nki/paged_attention.py",
                    },
                },
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
    stdx.debug.assert(q.rank() == 4 and q.shape().hasTags(.{ .b, .hkv, .hg, .hd }), "NKI paged_attention_2d expects q tags b,hkv,hg,hd, got {f}", .{q});
    stdx.debug.assert(k_cache.rank() == 4 and k_cache.shape().hasTags(.{ .page, .k_chunk, .hkv, .hd }), "NKI paged_attention_2d expects k_cache tags page,k_chunk,hkv,hd, got {f}", .{k_cache});
    stdx.debug.assert(v_cache.rank() == 4 and v_cache.shape().hasTags(.{ .page, .k_chunk, .hkv, .hd }), "NKI paged_attention_2d expects v_cache tags page,k_chunk,hkv,hd, got {f}", .{v_cache});
    stdx.debug.assert(k_cache.shape().eql(v_cache.shape()), "NKI paged_attention_2d expects matching K/V cache shapes, got {f} and {f}", .{ k_cache, v_cache });
    stdx.debug.assert(new_k.rank() == 3 and new_k.shape().hasTags(.{ .b, .hkv, .hd }), "NKI paged_attention_2d expects new_k tags b,hkv,hd, got {f}", .{new_k});
    stdx.debug.assert(new_v.shape().eql(new_k.shape()), "NKI paged_attention_2d expects matching new K/V shapes, got {f} and {f}", .{ new_k, new_v });
    stdx.debug.assert(new_k.dim(.b) == q.dim(.b), "NKI paged_attention_2d expects new K/V rows {} to match q rows {}", .{ new_k.dim(.b), q.dim(.b) });
    if (!parameters.options_.is_prefill) {
        stdx.debug.assert(slot_mapping != null, "NKI decode paged_attention_2d expects direct slot_mapping", .{});
        stdx.debug.assert(slot_mapping.?.rank() == 1 and slot_mapping.?.shape().hasTags(.{.b}), "NKI paged_attention_2d expects slot_mapping tag b, got {f}", .{slot_mapping.?});
        stdx.debug.assert(slot_mapping.?.dim(.b) == q.dim(.b), "NKI paged_attention_2d expects slot_mapping rows {} to match q rows {}", .{ slot_mapping.?.dim(.b), q.dim(.b) });
    }

    const block_table = parameters.block_table;
    const direct_slot_mapping = slot_mapping orelse parameters.query_start_len;
    const seq_lens = parameters.seq_lens
        .insertAxes(.last, .{.one});
    const query_start_len = parameters.query_start_len
        .insertAxes(.last, .{.one});

    return zml.ops.manualComputation(
        .{ q, k_cache, v_cache, new_k, new_v, direct_slot_mapping, block_table, seq_lens, query_start_len },
        q.shape(),
        parameters,
        (struct {
            fn body(kernel_options: Parameters, _: std.mem.Allocator, sharded_inputs: []const zml.Tensor, output: zml.Shape) zml.Tensor {
                stdx.debug.assert(sharded_inputs.len == 9, "NKI paged_attention_2d expects 9 sharded inputs, got {}", .{sharded_inputs.len});
                const kernel = kernel_options.options_.attention_kernel;
                return zml.ops.neuronNki(
                    .{ sharded_inputs[0], sharded_inputs[1], sharded_inputs[2], sharded_inputs[3], sharded_inputs[4], sharded_inputs[5], sharded_inputs[6], sharded_inputs[7], sharded_inputs[8] },
                    .{output},
                    .{
                        .name = kernel.name,
                        .entrypoint = kernel.entrypoint,
                        .source_path = kernel.source_path,
                        .compiler_target = kernel_options.options_.compiler_target,
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
    stdx.debug.assert(k_cache.rank() == 4 and k_cache.shape().hasTags(.{ .page, .k_chunk, .hkv, .hd }), "NKI paged KV update expects k_cache tags page,k_chunk,hkv,hd, got {f}", .{k_cache});
    stdx.debug.assert(v_cache.shape().eql(k_cache.shape()), "NKI paged KV update expects matching K/V cache shapes, got {f} and {f}", .{ k_cache, v_cache });
    stdx.debug.assert(new_k.rank() == 3 and new_k.shape().hasTags(.{ .b, .hkv, .hd }), "NKI paged KV update expects new_k tags b,hkv,hd, got {f}", .{new_k});
    stdx.debug.assert(new_v.shape().eql(new_k.shape()), "NKI paged KV update expects matching new K/V shapes, got {f} and {f}", .{ new_k, new_v });
    stdx.debug.assert(slot_mapping.rank() == 1 and slot_mapping.shape().hasTags(.{.b}), "NKI paged KV update expects slot_mapping tag b, got {f}", .{slot_mapping});
    stdx.debug.assert(slot_mapping.dim(.b) == new_k.dim(.b), "NKI paged KV update expects slot_mapping rows {} to match new K/V rows {}", .{ slot_mapping.dim(.b), new_k.dim(.b) });

    if (!parameters.options_.is_prefill) {
        return .{ .k = k_cache, .v = v_cache };
    }

    const query_start_len = parameters.query_start_len
        .insertAxes(.last, .{.one});

    const outputs = zml.ops.manualComputation(
        .{ k_cache, v_cache, new_k, new_v, slot_mapping, query_start_len },
        .{ k_cache.shape(), v_cache.shape() },
        parameters,
        (struct {
            fn body(kernel_options: Parameters, allocator: std.mem.Allocator, sharded_inputs: []const zml.Tensor, output: []const zml.Shape) []const zml.Tensor {
                stdx.debug.assert(sharded_inputs.len == 6, "NKI paged KV update expects 6 sharded inputs, got {}", .{sharded_inputs.len});
                stdx.debug.assert(output.len == 2, "NKI paged KV update expects 2 outputs, got {}", .{output.len});
                const kernel = kernel_options.options_.kv_update_kernel;
                const outputs_ = zml.ops.neuronNki(
                    .{ sharded_inputs[0], sharded_inputs[1], sharded_inputs[2], sharded_inputs[3], sharded_inputs[4], sharded_inputs[5] },
                    .{ output[0], output[1] },
                    .{
                        .name = kernel.name,
                        .entrypoint = kernel.entrypoint,
                        .source_path = kernel.source_path,
                        .compiler_target = kernel_options.options_.compiler_target,
                        .output_operand_aliases = &.{
                            .{ .output_index = 0, .operand_index = 0 },
                            .{ .output_index = 1, .operand_index = 1 },
                        },
                    },
                );
                const results = allocator.alloc(zml.Tensor, 2) catch unreachable;
                results[0] = outputs_[0];
                results[1] = outputs_[1];
                return results;
            }
        }).body,
    );

    return .{ .k = outputs[0], .v = outputs[1] };
}
