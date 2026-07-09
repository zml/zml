const std = @import("std");

const stdx = @import("stdx");

const zml = @import("../zml.zig");
const AttentionOptions = @import("paged_attention.zig").AttentionOptions;

const log = std.log.scoped(.@"zml/attention/metal");

pub const paged = struct {
    pub const Options = struct {
        batch_size: usize,
        max_num_pages: usize,
        max_seqlen_q: usize,
        is_prefill: bool,

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
            var allocation_size: usize = 0;

            allocation_size += self.block_table.byteSize();
            allocation_size += self.seq_lens.byteSize();
            allocation_size += self.query_start_len.byteSize();

            return allocation_size;
        }

        pub fn options(self: Parameters) Options {
            return self.options_;
        }

        pub fn onMemory(self: Parameters, memory: zml.platform.Memory.Kind) Parameters {
            return .{
                .options_ = self.options_,
                .block_table = self.block_table.onMemory(memory),
                .seq_lens = self.seq_lens.onMemory(memory),
                .query_start_len = self.query_start_len.onMemory(memory),
            };
        }

        pub fn toMemory(self: Parameters, memory: zml.platform.Memory.Kind) Parameters {
            return .{
                .options_ = self.options_,
                .block_table = self.block_table.toMemory(memory),
                .seq_lens = self.seq_lens.toMemory(memory),
                .query_start_len = self.query_start_len.toMemory(memory),
            };
        }
    };

    pub fn pagedAttention(parameters: Parameters, q: zml.Tensor, k_cache: zml.Tensor, v_cache: zml.Tensor, opts: AttentionOptions) zml.Tensor {
        const num_kv_heads = q.dim(.hkv);

        const qh = q.merge(.{ .h = .{ .hkv, .hg } }).transpose(.{ .b, .h, .hd });
        const kc = k_cache.transpose(.{ .page, .k_chunk, .hkv, .hd });
        const vc = v_cache.transpose(.{ .page, .k_chunk, .hkv, .hd });

        const out = zml.ops.customCall(
            "zml$paged_attn",
            .{ qh, kc, vc, parameters.block_table, parameters.seq_lens, parameters.query_start_len },
            qh.shape(),
            .{
                .is_causal = opts.is_causal,
                .sliding_window = opts.sliding_window,
                .scale = opts.scale,
            },
            .{ .has_side_effect = false },
        );

        // [total_q_tokens, num_heads, head_dim] -> {.b, .hkv, .hg, .hd} (q's order)
        return out.splitAxis(.h, .{ .hkv = num_kv_heads, .hg = .auto }).transpose(q.shape());
    }
};
