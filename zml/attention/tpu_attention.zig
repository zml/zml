const std = @import("std");

const mlir = @import("mlir");
const stdx = @import("stdx");

const ragged_paged = @import("platforms/tpu/ragged_paged");

const CompilationContext = @import("../module.zig").CompilationContext;
const zml = @import("../zml.zig");
const AttentionOptions = @import("paged_attention.zig").AttentionOptions;
const ragged_attention = @import("mosaic_tpu_kernels/ragged_attention.zig");

const log = std.log.scoped(.tpu_attention);

pub const mosaic_tpu = struct {
    const PreparedInputs = struct {
        q: zml.Tensor,
        kv_pages: zml.Tensor,
        seq_lens: zml.Tensor,
        block_table: zml.Tensor,
        query_start_len: zml.Tensor,
        num_seqs: zml.Tensor,
    };

    pub fn alignedKernelHeadDim(head_dim: usize) usize {
        stdx.debug.assert(head_dim > 0, "mosaic_tpu ragged paged attention head_dim must be non-zero", .{});
        const alignment: usize = 128;
        return (std.math.divCeil(usize, head_dim, alignment) catch unreachable) * alignment;
    }

    pub const Options = struct {
        is_prefill: bool,
        batch_size: usize,
        max_num_pages: usize,
        max_seqlen_k: usize,
        max_token_count: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        pages_per_compute_block: usize = 1,

        pub fn isPrefill(self: Options) bool {
            return self.is_prefill;
        }

        pub fn maxNumPages(self: Options) usize {
            return self.max_num_pages;
        }
    };

    pub const Parameters = struct {
        opts: Options,
        block_table: zml.Tensor,
        seq_lens: zml.Tensor,
        query_start_len: zml.Tensor,

        pub fn init(opts: Options) Parameters {
            return .{
                .opts = opts,
                .block_table = zml.Tensor.init(.{ .b = opts.batch_size, .p = opts.max_num_pages }, .i32),
                .seq_lens = zml.Tensor.init(.{ .b = opts.batch_size }, .i32),
                .query_start_len = zml.Tensor.init(.{ .b = opts.batch_size + 1 }, .i32),
            };
        }

        pub fn options(self: Parameters) Options {
            return self.opts;
        }

        pub fn allocationSize(self: Parameters) usize {
            var allocation_size: usize = 0;
            allocation_size += self.block_table.byteSize();
            allocation_size += self.seq_lens.byteSize();
            allocation_size += self.query_start_len.byteSize();
            return allocation_size;
        }
    };

    fn kernelFrontendAttrs(comp_ctx: *CompilationContext) *const mlir.Attribute {
        return mlir.dictionaryAttribute(comp_ctx.mlir_ctx, &.{
            mlir.NamedAttribute.named(comp_ctx.mlir_ctx, "kernel_metadata", mlir.stringAttribute(comp_ctx.mlir_ctx, "{}")),
        });
    }

    fn raggedPagedKernelCall(
        q: zml.Tensor,
        kv_pages: zml.Tensor,
        seq_lens: zml.Tensor,
        block_table: zml.Tensor,
        query_start_len: zml.Tensor,
        num_seqs: zml.Tensor,
        cfg: ragged_paged.Cfg,
    ) zml.Tensor {
        const seq_buf_idx = zml.Tensor.constant(zml.DataType.i32.zero()).broad(.init(.{2}, .i32));
        const comp_ctx = CompilationContext.current();

        const additional_attrs = [_]mlir.NamedAttribute{
            mlir.NamedAttribute.named(comp_ctx.mlir_ctx, "kernel_name", mlir.stringAttribute(comp_ctx.mlir_ctx, "ragged_paged_attention_kernel")),
            mlir.NamedAttribute.named(comp_ctx.mlir_ctx, "mhlo.frontend_attributes", kernelFrontendAttrs(comp_ctx)),
        };

        const out = ragged_attention.Kernel.call(
            .{
                .kv_lens = seq_lens,
                .page_indices = block_table,
                .cu_q_lens = query_start_len,
                .seq_buf_idx = seq_buf_idx,
                .num_seqs = num_seqs,
                .q = q,
                .kv_pages = kv_pages,
            },
            .{ .o = q.shape() },
            .{
                .cfg = cfg,
                .extras = .{
                    .vmem_limit_bytes = cfg.vmem_limit_bytes,
                    .additional_attributes = &additional_attrs,
                },
            },
        );
        return out.o;
    }

    fn activeSequenceCount(query_start_len: zml.Tensor) zml.Tensor {
        const start = query_start_len.slice1d(.b, .{ .end = query_start_len.dim(.b) - 1 });
        const end = query_start_len.slice1d(.b, .{ .start = 1 });
        const query_lens = end.sub(start);
        return query_lens
            .cmp(.GT, zml.Tensor.constant(query_lens.dtype().zero()).broad(query_lens.shape()))
            .convert(.i32)
            .sum(.b)
            .reshape(zml.Shape.init(.{1}, .i32));
    }

    fn cfgDtype(dtype: zml.DataType) @FieldType(ragged_paged.Cfg, "q_dtype") {
        const cast = zml.kernel.mosaic_tpu.from(dtype);
        return switch (cast) {
            .bf16, .f16, .f32 => cast,
            else => stdx.debug.panic("mosaic_tpu ragged paged attention expects q/kv dtype bf16/f16/f32, got {}", .{dtype}),
        };
    }

    inline fn restoreQueryHeads(q_template: zml.Tensor, q_out: zml.Tensor) zml.Tensor {
        const restored = q_out.splitAxis(.h, .{ .hkv = q_template.dim(.hkv), .hg = q_template.dim(.hg) });
        if (restored.dim(.hd) == q_template.dim(.hd)) return restored;
        stdx.debug.assert(
            restored.dim(.hd) > q_template.dim(.hd),
            "mosaic_tpu ragged paged attention cannot restore output from head_dim {} to {}",
            .{ restored.dim(.hd), q_template.dim(.hd) },
        );
        return restored.slice1d(.hd, .{ .end = q_template.dim(.hd) });
    }

    inline fn alignQueryHeadDimForKernel(q: zml.Tensor, target_head_dim: i64) zml.Tensor {
        if (q.dim(.hd) == target_head_dim) return q;
        stdx.debug.assert(
            q.dim(.hd) < target_head_dim,
            "mosaic_tpu ragged paged attention q head_dim {} exceeds kernel/cache head_dim {}",
            .{ q.dim(.hd), target_head_dim },
        );
        return q.pad(0, .{ .hd = zml.Tensor.Pad{ .high = target_head_dim - q.dim(.hd) } })
            .withPartitioning(.{ .h = .model });
    }

    inline fn prepareInputs(parameters: Parameters, q: zml.Tensor, kv_pages: zml.Tensor) PreparedInputs {
        const kv_pages_partitioned = kv_pages.withPartitioning(.{ .hkv = .model });
        const q_merged = q.merge(.{ .h = .{ .hkv, .hg } }).withPartitioning(.{ .h = .model });
        const logical_head_dim: usize = @intCast(q_merged.dim(.hd));
        const cache_head_dim: usize = @intCast(kv_pages_partitioned.dim(.hd));
        stdx.debug.assert(
            cache_head_dim == alignedKernelHeadDim(logical_head_dim),
            "mosaic_tpu ragged paged attention expects KV cache head_dim {} to match aligned logical head_dim {}",
            .{ cache_head_dim, logical_head_dim },
        );
        const q_ragged = alignQueryHeadDimForKernel(q_merged, kv_pages_partitioned.dim(.hd));
        const query_start_len = parameters.query_start_len.withPartitioning(.{ .b = .replicated });

        return .{
            .q = q_ragged,
            .kv_pages = kv_pages_partitioned,
            .seq_lens = parameters.seq_lens.withPartitioning(.{ ._0 = .replicated }),
            .block_table = parameters.block_table.withPartitioning(.{ ._0 = .replicated, ._1 = .replicated }),
            .query_start_len = query_start_len,
            .num_seqs = activeSequenceCount(query_start_len).withPartitioning(.{ ._0 = .replicated }),
        };
    }

    fn buildCfg(prepared_inputs: PreparedInputs, parameters: Parameters, opts: AttentionOptions) ragged_paged.Cfg {
        const q_shape = prepared_inputs.q.shape();
        const kv_pages_shape = prepared_inputs.kv_pages.shape();
        const block_table_shape = prepared_inputs.block_table.shape();

        const q_token_count = if (q_shape.hasTag(.q) != null) q_shape.dim(.q) else q_shape.dim(.b);
        const max_num_seqs = block_table_shape.dim(.b);
        const num_q_heads = q_shape.dim(.h);
        stdx.debug.assert(
            q_shape.dim(.hd) == kv_pages_shape.dim(.hd),
            "mosaic_tpu ragged paged attention expects q and KV cache head_dim to match, got {} and {}",
            .{ q_shape.dim(.hd), kv_pages_shape.dim(.hd) },
        );
        stdx.debug.assert(@mod(kv_pages_shape.dim(.hkv), 2) == 0, "Expected fused KV pages .hkv dimension to be even, got {}", .{kv_pages_shape.dim(.hkv)});
        const num_kv_heads = std.math.divExact(i64, kv_pages_shape.dim(.hkv), 2) catch
            stdx.debug.panic("Expected fused KV pages .hkv dimension to be even, got {}", .{kv_pages_shape.dim(.hkv)});
        const num_queries_per_block = @max(@as(i64, 1), @min(q_token_count, max_num_seqs));
        const logical_head_dim: f64 = @floatFromInt(parameters.opts.head_dim);

        return .{
            .num_q_tokens = q_token_count,
            .num_q_heads = num_q_heads,
            .num_kv_heads = num_kv_heads,
            .head_dim = q_shape.dim(.hd),
            .total_num_pages = kv_pages_shape.dim(.page),
            .page_size = kv_pages_shape.dim(.k_chunk),
            .max_num_seqs = max_num_seqs,
            .pages_per_seq = @intCast(parameters.opts.max_num_pages),
            .num_kv_pages_per_block = @intCast(parameters.opts.pages_per_compute_block),
            .num_queries_per_block = num_queries_per_block,
            .q_dtype = cfgDtype(q_shape.dtype()),
            .kv_dtype = cfgDtype(kv_pages_shape.dtype()),
            .sm_scale = opts.scale orelse @floatCast(1.0 / @sqrt(logical_head_dim)),
            .sliding_window = if (opts.sliding_window < 0) null else @intCast(opts.sliding_window),
        };
    }

    pub const Context = struct {
        pub fn init(parameters: Parameters, num_heads: i64, num_kv_heads: i64, head_dim: i64, page_size: i64) Context {
            _ = parameters;
            _ = num_heads;
            _ = num_kv_heads;
            _ = head_dim;
            _ = page_size;
            return .{};
        }
    };

    pub fn pagedAttention(parameters: Parameters, context: Context, q: zml.Tensor, kv_cache: zml.Tensor, opts: AttentionOptions) zml.Tensor {
        stdx.debug.assert(opts.is_causal, "mosaic_tpu ragged paged attention currently only supports causal attention", .{});

        const prepared = prepareInputs(parameters, q, kv_cache);

        const q_out = zml.ops.manualComputation(
            .{
                prepared.q,
                prepared.kv_pages,
                prepared.seq_lens,
                prepared.block_table,
                prepared.query_start_len,
                prepared.num_seqs,
            },
            prepared.q.shape(),
            .{
                .opts = opts,
                .parameters = parameters,
                .context = context,
            },
            (struct {
                fn body(body_context: anytype, allocator: std.mem.Allocator, sharded_inputs: []const zml.Tensor, output: zml.Shape) zml.Tensor {
                    _ = allocator;
                    stdx.debug.assert(sharded_inputs.len == 6, "mosaic_tpu ragged paged manualComputation expects 6 inputs, got {}", .{sharded_inputs.len});

                    const prepared_inputs: PreparedInputs = .{
                        .q = sharded_inputs[0],
                        .kv_pages = sharded_inputs[1],
                        .seq_lens = sharded_inputs[2],
                        .block_table = sharded_inputs[3],
                        .query_start_len = sharded_inputs[4],
                        .num_seqs = sharded_inputs[5],
                    };

                    const cfg = buildCfg(prepared_inputs, body_context.parameters, body_context.opts);
                    const q_out = raggedPagedKernelCall(
                        prepared_inputs.q,
                        prepared_inputs.kv_pages,
                        prepared_inputs.seq_lens,
                        prepared_inputs.block_table,
                        prepared_inputs.query_start_len,
                        prepared_inputs.num_seqs,
                        cfg,
                    );
                    stdx.debug.assert(q_out.shape().eql(output), "mosaic_tpu ragged paged attention output shape mismatch, got {f}, expected {f}", .{ q_out.shape(), output });
                    return q_out;
                }
            }).body,
        );

        const restored = restoreQueryHeads(q, q_out);
        stdx.debug.assert(restored.shape().eql(q.shape()), "mosaic_tpu ragged paged attention output shape mismatch, got {f}, expected {f}", .{ restored.shape(), q.shape() });
        return restored;
    }
};

test "mosaic_tpu aligns kernel head dimensions to 128" {
    try std.testing.expectEqual(@as(usize, 128), mosaic_tpu.alignedKernelHeadDim(64));
    try std.testing.expectEqual(@as(usize, 128), mosaic_tpu.alignedKernelHeadDim(128));
    try std.testing.expectEqual(@as(usize, 256), mosaic_tpu.alignedKernelHeadDim(180));
    try std.testing.expectEqual(@as(usize, 256), mosaic_tpu.alignedKernelHeadDim(256));
}

test "mosaic_tpu cfg uses kernel head dim and logical scale" {
    const parameters = mosaic_tpu.Parameters.init(.{
        .is_prefill = false,
        .batch_size = 1,
        .max_num_pages = 2,
        .max_seqlen_k = 32,
        .max_token_count = 1,
        .num_heads = 8,
        .num_kv_heads = 2,
        .head_dim = 64,
    });
    const prepared_inputs: mosaic_tpu.PreparedInputs = .{
        .q = zml.Tensor.init(.{ .b = 1, .h = 8, .hd = 128 }, .bf16),
        .kv_pages = zml.Tensor.init(.{ .page = 2, .k_chunk = 16, .hkv = 4, .hd = 128 }, .bf16),
        .seq_lens = parameters.seq_lens,
        .block_table = parameters.block_table,
        .query_start_len = parameters.query_start_len,
        .num_seqs = zml.Tensor.init(.{1}, .i32),
    };

    const cfg = mosaic_tpu.buildCfg(prepared_inputs, parameters, .{});
    try std.testing.expectEqual(@as(i64, 128), cfg.head_dim);
    try std.testing.expectApproxEqAbs(
        @as(f32, 1.0 / @sqrt(@as(f32, 64.0))),
        cfg.sm_scale,
        0.000001,
    );
}
