const std = @import("std");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const dialects = @import("mlir/dialects");
const mlir = @import("mlir");
const stdx = @import("stdx");

const CompilationContext = @import("../module.zig").CompilationContext;
const zml = @import("../zml.zig");
const AttentionOptions = @import("paged_attention.zig").AttentionOptions;

const log = std.log.scoped(.tpu_attention);

pub const Runtime = struct {
    process: std.process.Child,
    request_mutex: std.Io.Mutex = .init,

    fn getGenerateBinPath(allocator: std.mem.Allocator) ![]const u8 {
        const runfiles = try bazel.runfiles(bazel_builtin.current_repository);

        var sandbox_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const sandbox_path = try runfiles.rlocation("libpjrt_tpu/sandbox", &sandbox_path_buf) orelse return error.FileNotFound;

        return std.fs.path.join(allocator, &.{ sandbox_path, "bin", "gen_ir_zig" });
    }

    pub fn init(allocator: std.mem.Allocator, io: std.Io) !Runtime {
        const path = try getGenerateBinPath(allocator);
        defer allocator.free(path);

        var process = try std.process.spawn(io, .{
            .argv = &.{path},
            .stdin = .pipe,
            .stdout = .pipe,
        });
        errdefer process.kill(io);

        return .{ .process = process };
    }

    pub fn request(self: *Runtime, allocator: std.mem.Allocator, io: std.Io, request_json: []const u8) ![]u8 {
        self.request_mutex.lockUncancelable(io);
        defer self.request_mutex.unlock(io);

        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();

        const writer_buffer = try arena.allocator().alloc(u8, 4096);
        var writer = self.process.stdin.?.writer(io, writer_buffer);
        try writer.interface.print("{s}\n", .{request_json});
        try writer.interface.flush();

        const reader_buffer = try arena.allocator().alloc(u8, 4096);
        var reader = self.process.stdout.?.reader(io, reader_buffer);
        var allocating: std.Io.Writer.Allocating = .init(arena.allocator());
        _ = try reader.interface.streamDelimiter(&allocating.writer, '\n');
        _ = try reader.interface.discardShort(1);

        const response: std.json.Value = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), allocating.written(), .{});
        const response_object = switch (response) {
            .object => |object| object,
            else => {
                log.err("TPU backend config response is not a JSON object: {s}", .{allocating.written()});
                return error.InvalidResponse;
            },
        };
        const ok = response_object.get("ok") orelse {
            log.err("TPU backend config response is missing `ok`: {s}", .{allocating.written()});
            return error.InvalidResponse;
        };
        const ok_bool = switch (ok) {
            .bool => |value| value,
            else => {
                log.err("TPU backend config response has non-bool `ok`: {s}", .{allocating.written()});
                return error.InvalidResponse;
            },
        };

        if (ok_bool) {
            const result = response_object.get("result") orelse {
                log.err("TPU backend config response is missing `result`: {s}", .{allocating.written()});
                return error.InvalidResponse;
            };
            return switch (result) {
                .string => |value| try allocator.dupe(u8, value),
                else => {
                    log.err("TPU backend config response has non-string `result`: {s}", .{allocating.written()});
                    return error.InvalidResponse;
                },
            };
        }

        if (response_object.get("error")) |value| {
            switch (value) {
                .string => |message| log.err("TPU backend config generation failed: {s}", .{message}),
                else => log.err("TPU backend config generation failed with invalid error payload: {s}", .{allocating.written()}),
            }
        } else {
            log.err("TPU backend config generation failed without an error payload: {s}", .{allocating.written()});
        }
        return error.GenerationFailed;
    }

    pub fn deinit(self: *Runtime, io: std.Io) void {
        _ = std.c.kill(self.process.id.?, .INT);
        _ = self.process.wait(io) catch unreachable;
    }
};

pub const mosaic_tpu = struct {
    const RaggedPagedRequest = struct {
        num_q_tokens: i64,
        num_q_heads: i64,
        num_kv_heads: i64,
        head_dim: i64,
        q_dtype: []const u8,
        kv_dtype: []const u8,
        max_num_seqs: i64,
        pages_per_seq: i64,
        total_num_pages: i64,
        page_size: i64,
        sm_scale: f32,
        sliding_window: ?i64 = null,
        num_kv_pages_per_block: u32,
        num_queries_per_block: u32,
    };

    const PreparedInputs = struct {
        q: zml.Tensor,
        kv_pages: zml.Tensor,
        seq_lens: zml.Tensor,
        block_table: zml.Tensor,
        query_start_len: zml.Tensor,
        num_seqs: zml.Tensor,
    };

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
                .block_table = zml.Tensor.init(.{ opts.batch_size, opts.max_num_pages }, .i32),
                .seq_lens = zml.Tensor.init(.{opts.batch_size}, .i32),
                .query_start_len = zml.Tensor.init(.{ .n = opts.batch_size + 1 }, .i32),
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
        backend_config: []const u8,
    ) zml.Tensor {
        const seq_buf_idx = zml.Tensor.constant(zml.DataType.i32.zero()).broad(.init(.{2}, .i32));

        const comp_ctx = CompilationContext.current();
        const op = dialects.stablehlo.custom_call(
            comp_ctx.mlir_ctx,
            &.{
                seq_lens.value(),
                block_table.value(),
                query_start_len.value(),
                seq_buf_idx.value(),
                num_seqs.value(),
                q.value(),
                kv_pages.value(),
            },
            &.{q.value().type_()},
            .{
                .call_target_name = "tpu_custom_call",
                .backend_config = .{ .original = backend_config },
                .has_side_effect = false,
                .additional_attributes = &.{
                    mlir.NamedAttribute.named(comp_ctx.mlir_ctx, "kernel_name", mlir.stringAttribute(comp_ctx.mlir_ctx, "ragged_paged_attention_kernel")),
                    mlir.NamedAttribute.named(comp_ctx.mlir_ctx, "mhlo.frontend_attributes", kernelFrontendAttrs(comp_ctx)),
                },
            },
            .unknown(comp_ctx.mlir_ctx),
        ).appendTo(comp_ctx.currentScope().block);
        return zml.Tensor._result(q.shape(), op.result(0));
    }

    fn activeSequenceCount(query_start_len: zml.Tensor) zml.Tensor {
        const start = query_start_len.slice1d(0, .{ .end = query_start_len.dim(0) - 1 });
        const end = query_start_len.slice1d(0, .{ .start = 1 });
        const query_lens = end.sub(start);
        return query_lens
            .cmp(.GT, zml.Tensor.constant(query_lens.dtype().zero()).broad(query_lens.shape()))
            .convert(.i32)
            .sum(0)
            .reshape(zml.Shape.init(.{1}, .i32));
    }

    fn jnpDTypeExpr(dtype: zml.DataType) []const u8 {
        return switch (dtype) {
            .bf16 => "jnp.bfloat16",
            .f32 => "jnp.float32",
            else => stdx.debug.panic("mosaic_tpu ragged paged attention expects q/kv dtype bf16 or f32, got {}", .{dtype}),
        };
    }

    fn renderBackendConfig(allocator: std.mem.Allocator, io: std.Io, request: RaggedPagedRequest) []u8 {
        const compilation_context = CompilationContext.current();

        const json_string = std.fmt.allocPrint(
            allocator,
            "{{\"backend_config\":\"ragged_paged\",\"params\":{f}}}",
            .{std.json.fmt(request, .{})},
        ) catch |err| stdx.debug.panic("Failed to allocate ragged paged attention TPU params: {}", .{err});
        defer allocator.free(json_string);

        const platform = @constCast(compilation_context.platform);
        const runtime = &(platform.tpu_ir_runtime orelse stdx.debug.panic("TPU backend config runtime is not initialized", .{}));
        const backend_config = runtime.request(allocator, io, json_string) catch |err| {
            stdx.debug.panic("Failed to generate TPU backend config through persistent runtime: {}", .{err});
        };
        defer allocator.free(backend_config);

        return allocator.dupe(u8, backend_config) catch |err| {
            stdx.debug.panic("Failed to persist TPU backend config in compilation arena: {}", .{err});
        };
    }

    inline fn restoreQueryHeads(q_template: zml.Tensor, q_out: zml.Tensor) zml.Tensor {
        return q_out.splitAxis(.h, .{ .hkv = q_template.dim(.hkv), .hg = q_template.dim(.hg) });
    }

    inline fn fuseKvPages(k_cache: zml.Tensor, v_cache: zml.Tensor) zml.Tensor {
        const k_pages = k_cache.insertAxes(.hd, .{.kv});
        const v_pages = v_cache.insertAxes(.hd, .{.kv});
        return zml.Tensor.concatenate(&.{ k_pages, v_pages }, .kv)
            .merge(.{ .hkv = .{ .hkv, .kv } });
    }

    inline fn prepareInputs(parameters: Parameters, q: zml.Tensor, k_cache: zml.Tensor, v_cache: zml.Tensor) PreparedInputs {
        stdx.debug.assert(k_cache.shape().eqlWithTags(v_cache.shape()), "Expected paged K/V cache shapes to match, got {f} and {f}", .{ k_cache.shape(), v_cache.shape() });

        const q_ragged = q.merge(.{ .h = .{ .hkv, .hg } }).withPartitioning(.{ .h = .model });
        // Note :
        // In a future PR, we'll have to change the pagedAttention API a bit so that we don't have to fuse keys and values
        const kv_pages = fuseKvPages(k_cache, v_cache).withPartitioning(.{ .hkv = .model });
        const query_start_len = parameters.query_start_len.withPartitioning(.{ .n = .replicated });

        // Note
        // activeSequenceCount could probably be computed when preparing the inputs and injected through Parameters, but we'll keep that for later.
        return .{
            .q = q_ragged,
            .kv_pages = kv_pages,
            .seq_lens = parameters.seq_lens.withPartitioning(.{ ._0 = .replicated }),
            .block_table = parameters.block_table.withPartitioning(.{ ._0 = .replicated, ._1 = .replicated }),
            .query_start_len = query_start_len,
            .num_seqs = activeSequenceCount(query_start_len).withPartitioning(.{ ._0 = .replicated }),
        };
    }

    fn checkedU32(value: usize, comptime field_name: []const u8) u32 {
        stdx.debug.assert(value <= std.math.maxInt(u32), "mosaic_tpu ragged paged attention field '{s}' exceeds u32 range: {}", .{ field_name, value });
        return @intCast(value);
    }

    fn buildRequest(prepared_inputs: PreparedInputs, parameters: Parameters, opts: AttentionOptions) RaggedPagedRequest {
        const q_shape = prepared_inputs.q.shape();
        const kv_pages_shape = prepared_inputs.kv_pages.shape();
        const block_table_shape = prepared_inputs.block_table.shape();

        const q_token_count = if (q_shape.hasTag(.q) != null) q_shape.dim(.q) else q_shape.dim(.b);
        const max_num_seqs = block_table_shape.dim(0);
        const num_q_heads = q_shape.dim(.h);
        stdx.debug.assert(@mod(kv_pages_shape.dim(.hkv), 2) == 0, "Expected fused KV pages .hkv dimension to be even, got {}", .{kv_pages_shape.dim(.hkv)});
        const num_kv_heads = std.math.divExact(i64, kv_pages_shape.dim(.hkv), 2) catch |err| switch (err) {
            error.UnexpectedRemainder => stdx.debug.panic("Expected fused KV pages .hkv dimension to be even, got {}", .{kv_pages_shape.dim(.hkv)}),
            error.DivisionByZero => stdx.debug.panic("KV pages .hkv dimension must be non-zero, got {}", .{kv_pages_shape.dim(.hkv)}),
            error.Overflow => stdx.debug.panic("KV pages .hkv dimension is too large to fit in i64: {}", .{kv_pages_shape.dim(.hkv)}),
        };
        const num_queries_per_block = @max(@as(i64, 1), @min(q_token_count, max_num_seqs));
        const sm_scale: f32 = opts.scale orelse @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(q_shape.dim(.hd)))));

        return .{
            .num_q_tokens = q_token_count,
            .num_q_heads = num_q_heads,
            .num_kv_heads = num_kv_heads,
            .head_dim = q_shape.dim(.hd),
            .q_dtype = jnpDTypeExpr(q_shape.dtype()),
            .kv_dtype = jnpDTypeExpr(kv_pages_shape.dtype()),
            .max_num_seqs = max_num_seqs,
            .pages_per_seq = @intCast(parameters.opts.max_num_pages),
            .total_num_pages = kv_pages_shape.dim(.page),
            .page_size = kv_pages_shape.dim(.k_chunk),
            .sm_scale = sm_scale,
            .sliding_window = if (opts.sliding_window < 0) null else @intCast(opts.sliding_window),
            .num_kv_pages_per_block = checkedU32(parameters.opts.pages_per_compute_block, "num_kv_pages_per_block"),
            .num_queries_per_block = checkedU32(@intCast(num_queries_per_block), "num_queries_per_block"),
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

    pub fn pagedAttention(parameters: Parameters, context: Context, q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, k_cache: zml.Tensor, v_cache: zml.Tensor, opts: AttentionOptions) zml.Tensor {
        _ = k;
        _ = v;
        stdx.debug.assert(opts.is_causal, "mosaic_tpu ragged paged attention currently only supports causal attention", .{});

        const prepared = prepareInputs(parameters, q, k_cache, v_cache);

        // Keep sharding intent on the `manualComputation` boundary only. The
        // TPU smoke compile regressed when the body restated
        // `withPartitioning` constraints on intermediate tensors.
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
                    stdx.debug.assert(sharded_inputs.len == 6, "mosaic_tpu ragged paged manualComputation expects 6 inputs, got {}", .{sharded_inputs.len});

                    const prepared_inputs: PreparedInputs = .{
                        .q = sharded_inputs[0],
                        .kv_pages = sharded_inputs[1],
                        .seq_lens = sharded_inputs[2],
                        .block_table = sharded_inputs[3],
                        .query_start_len = sharded_inputs[4],
                        .num_seqs = sharded_inputs[5],
                    };

                    const backend_config = renderBackendConfig(allocator, CompilationContext.current().io, buildRequest(prepared_inputs, body_context.parameters, body_context.opts));
                    const q_out = raggedPagedKernelCall(
                        prepared_inputs.q,
                        prepared_inputs.kv_pages,
                        prepared_inputs.seq_lens,
                        prepared_inputs.block_table,
                        prepared_inputs.query_start_len,
                        prepared_inputs.num_seqs,
                        backend_config,
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
