const std = @import("std");

const flashattn = @import("platforms/cuda/flashattn");
const platforms = @import("platforms");
const stdx = @import("stdx");

const CompilationContext = @import("../module.zig").CompilationContext;
const zml = @import("../zml.zig");
const ffi = zml.pjrt.ffi;
const AttentionOptions = @import("paged_attention.zig").AttentionOptions;

const log = std.log.scoped(.@"zml/attention/flashattn");

pub fn load(allocator: std.mem.Allocator, io: std.Io) !void {
    if (comptime platforms.isEnabled(.cuda)) {
        try flashattn.load(allocator, io);
    }
}

pub fn register(platform: *const zml.Platform) !void {
    if (comptime platforms.isEnabled(.cuda)) {
        try fa2.register(platform);
        try fa3.register(platform);
        try paged_fa2.Decode.register(platform);
        try paged_fa2.Prefill.register(platform);
        try paged_fa3.Decode.register(platform);
        try paged_fa3.Prefill.register(platform);
    }
}

fn flashattnDataTypeFromZmlDataType(dtype: zml.DataType) flashattn.DataType {
    return switch (dtype) {
        .f32 => .f32,
        .f16 => .f16,
        .bf16 => .bf16,
        .i32 => .i32,
        .i8 => .i8,
        else => unreachable,
    };
}

const Buffer = struct {
    shape: zml.Shape,
    ptr: *anyopaque,

    fn toFlashattnTensor(buffer: Buffer) flashattn.Tensor {
        return .init(
            buffer.ptr,
            buffer.shape.dims(),
            buffer.shape.withDtype(.u8).computeByteStrides().constSlice(),
            flashattnDataTypeFromZmlDataType(buffer.shape.dtype()),
        );
    }

    pub fn format(
        self: @This(),
        writer: *std.Io.Writer,
    ) std.Io.Writer.Error!void {
        try writer.print("{f}@{*}", .{ self.shape, self.ptr });
    }
};

fn getPlatform(call_frame: *ffi.CallFrame) zml.Platform {
    const pjrt_api_ptr = call_frame.attrs.getByName(.scalar, "pjrt_api") orelse unreachable;
    std.debug.assert(pjrt_api_ptr.dtype == .u64);
    const pjrt_api: ?*zml.pjrt.Api = @ptrFromInt(pjrt_api_ptr.get(usize));

    const pjrt_client_ptr = call_frame.attrs.getByName(.scalar, "pjrt_client") orelse unreachable;
    std.debug.assert(pjrt_client_ptr.dtype == .u64);
    const pjrt_client: ?*zml.pjrt.Client = @ptrFromInt(pjrt_client_ptr.get(usize));

    return .{ .target = .cuda, .pjrt_api = pjrt_api.?, .pjrt_client = pjrt_client.? };
}

fn dataTypeFromFfiDataType(ffi_dt: ffi.DataType) zml.DataType {
    return switch (ffi_dt) {
        .bool => .bool,
        .i8 => .i8,
        .i16 => .i16,
        .i32 => .i32,
        .i64 => .i64,
        .u8 => .u8,
        .u16 => .u16,
        .u32 => .u32,
        .u64 => .u64,
        .f16 => .f16,
        .f32 => .f32,
        .f64 => .f64,
        .bf16 => .bf16,
        .c64 => .c64,
        .c128 => .c128,
        .f8e5m2 => .f8e5m2,
        .f8e4m3fn => .f8e4m3fn,
        .f8e4m3b11fnuz => .f8e4m3b11fnuz,
        .f8e5m2fnuz => .f8e5m2fnuz,
        .f8e4m3fnuz => .f8e4m3fnuz,
        else => unreachable,
    };
}

fn shapeFromFfiBuffer(buffer: *const ffi.Buffer) zml.Shape {
    return .init(buffer.dims(), dataTypeFromFfiDataType(buffer.dtype));
}

fn bufferFromFfiBuffer(ffi_buffer: *const ffi.Buffer) Buffer {
    return .{
        .shape = shapeFromFfiBuffer(ffi_buffer),
        .ptr = ffi_buffer.data,
    };
}

fn getScalarAttributeAs(comptime T: type, call_frame: *ffi.CallFrame, attribute_name: []const u8) ?T {
    const attribute = call_frame.attrs.getByName(.scalar, attribute_name) orelse return null;
    return attribute.get(T);
}

fn fixupKvCacheBuffer(buffer: Buffer, layer_index: i64) Buffer {
    var shape = buffer.shape;
    const layer_stride = shape.computeByteStrides().get(0);
    shape = shape.remove(0);
    const ptr = @as([*]u8, @ptrCast(buffer.ptr));
    return .{
        .shape = shape,
        .ptr = ptr + @as(usize, @intCast(layer_stride * layer_index)),
    };
}

fn pagedCacheToDenseForLayer(
    cache: zml.Tensor,
    layer_index: zml.Tensor,
    block_table: zml.Tensor,
    max_seqlen_k: usize,
) zml.Tensor {
    const layer_coord = layer_index.reshape(.{1}).withTags(.{.layer_sel});
    const layer_cache = cache.gather(.{ .layer = layer_coord }, .{}).squeeze(.layer_sel);
    const page_size = layer_cache.dim(.k_chunk);
    const page_count = layer_cache.dim(.page);
    const key_count = @min(max_seqlen_k, page_count * page_size);

    const key_pos = zml.Tensor.iota(zml.Shape.init(.{ .tk = key_count }, .i32), .tk);
    const page_index = key_pos.divByConst(page_size);
    const chunk_index = key_pos.remainder(zml.Tensor.constant(key_pos.dtype().constant(page_size)).broad(key_pos.shape()));

    var pages = block_table.gather(.{ .p = page_index }, .{});
    const pages_min = zml.Tensor.constant(pages.dtype().zero()).broad(pages.shape());
    const pages_max = zml.Tensor.constant(pages.dtype().constant(page_count - 1)).broad(pages.shape());
    pages = pages.maximum(pages_min).minimum(pages_max);

    const page_stride = zml.Tensor.constant(pages.dtype().constant(page_size)).broad(pages.shape());
    const slot_index = pages.mul(page_stride).add(chunk_index.broad(pages.shape()));

    const flat_cache = layer_cache.merge(.{ .k = .{ .page, .k_chunk } });
    return flat_cache.gather(.{ .k = slot_index }, .{}).withTags(.{ .s, .k, .hkv, .hd });
}

fn pagedAttentionVanillaGroup(
    q_: zml.Tensor,
    k_cache: zml.Tensor,
    v_cache: zml.Tensor,
    layer_index: zml.Tensor,
    block_table_: zml.Tensor,
    cu_seqlens_q_: zml.Tensor,
    seqused_k_: zml.Tensor,
    max_seqlen_k: usize,
    opts: AttentionOptions,
) zml.Tensor {
    const q = q_.withTags(.{ .b, .hkv, .hg, .hd });
    const block_table = block_table_.withTags(.{ .s, .p });
    const cu_seqlens_q = cu_seqlens_q_.withTags(.{ .s });
    const seqused_k = seqused_k_.withTags(.{ .s });

    const seq_count = block_table.dim(.s);
    const q_count = q.dim(.b);
    const key_count = @min(max_seqlen_k, block_table.dim(.p) * k_cache.dim(.k_chunk));
    const num_kv_heads = q.dim(.hkv);
    const num_head_groups = q.dim(.hg);

    const k_dense = pagedCacheToDenseForLayer(k_cache, layer_index, block_table, max_seqlen_k);
    const v_dense = pagedCacheToDenseForLayer(v_cache, layer_index, block_table, max_seqlen_k);

    const q_pos = zml.Tensor.iota(zml.Shape.init(.{ .b = q_count }, .i32), .b);
    const shape_sb = zml.Shape.init(.{ .s = seq_count, .b = q_count }, .bool);
    const shape_sk = zml.Shape.init(.{ .s = seq_count, .k = key_count }, .bool);
    const shape_sbk = zml.Shape.init(.{ .s = seq_count, .b = q_count, .k = key_count }, .bool);
    const shape_sbhg = zml.Shape.init(.{
        .s = seq_count,
        .b = q_count,
        .hkv = num_kv_heads,
        .hg = num_head_groups,
        .hd = q.dim(.hd),
    }, q.dtype());

    const cu_start = cu_seqlens_q.slice1d(.s, .{ .end = seq_count }).withTags(.{ .s });
    const cu_end = cu_seqlens_q.slice1d(.s, .{ .start = 1 }).withTags(.{ .s });
    const query_len = cu_end.sub(cu_start);

    const q_pos_sb = q_pos.broad(shape_sb);
    const cu_start_sb = cu_start.broad(shape_sb);
    const cu_end_sb = cu_end.broad(shape_sb);

    var valid_q = q_pos_sb.cmp(.GE, cu_start_sb);
    valid_q = valid_q.logical(.AND, q_pos_sb.cmp(.LT, cu_end_sb));

    var q_by_seq = q.broad(shape_sbhg);
    const q_zero = zml.Tensor.constant(q.dtype().zero()).broad(shape_sbhg);
    q_by_seq = zml.Tensor.select(valid_q.broad(shape_sbhg), q_by_seq, q_zero);

    const k_pos = zml.Tensor.iota(zml.Shape.init(.{ .k = key_count }, .i32), .k);
    const valid_k = k_pos.broad(shape_sk).cmp(.LT, seqused_k.broad(shape_sk));

    const q_base = seqused_k.sub(query_len).broad(shape_sb);
    const q_rel_pos = q_pos_sb.sub(cu_start_sb);
    const q_abs_pos = q_base.add(q_rel_pos);

    var mask_sqk = valid_q.broad(shape_sbk).logical(.AND, valid_k.broad(shape_sbk));
    if (opts.is_causal) {
        var causal_mask = k_pos.broad(shape_sbk).cmp(.LE, q_abs_pos.broad(shape_sbk));
        if (opts.sliding_window >= 0) {
            const q_window_start = q_abs_pos.addConstant(-opts.sliding_window).broad(shape_sbk);
            const window_mask = k_pos.broad(shape_sbk).cmp(.GE, q_window_start);
            causal_mask = causal_mask.logical(.AND, window_mask);
        }
        mask_sqk = mask_sqk.logical(.AND, causal_mask);
    } else if (opts.sliding_window >= 0) {
        const q_window_start = q_abs_pos.addConstant(-opts.sliding_window).broad(shape_sbk);
        const window_mask = k_pos.broad(shape_sbk).cmp(.GE, q_window_start);
        mask_sqk = mask_sqk.logical(.AND, window_mask);
    }

    const mask_shape = zml.Shape.init(.{
        .s = seq_count,
        .hkv = num_kv_heads,
        .b = q_count,
        .k = key_count,
    }, .bool);
    var attn_mask = mask_sqk.broad(mask_shape).merge(.{ .h = .{ .s, .hkv } }).withTags(.{ .h, .q, .k });
    const attn_zero = zml.Tensor.constant(q.dtype().zero()).broad(attn_mask.shape());
    const attn_minus_inf = zml.Tensor.constant(q.dtype().minValue()).broad(attn_mask.shape());
    attn_mask = zml.Tensor.select(attn_mask, attn_zero, attn_minus_inf);

    const q_sdpa = q_by_seq.transpose(.{ .s, .hkv, .hg, .b, .hd }).merge(.{ .h = .{ .s, .hkv, .hg } }).withTags(.{ .h, .q, .hd }).withPartitioning(.{ .h = .model });
    const k_sdpa = k_dense.transpose(.{ .s, .hkv, .k, .hd }).merge(.{ .h = .{ .s, .hkv } }).withTags(.{ .h, .k, .hd }).withPartitioning(.{ .h = .model });
    const v_sdpa = v_dense.transpose(.{ .s, .hkv, .k, .hd }).merge(.{ .h = .{ .s, .hkv } }).withTags(.{ .h, .k, .hd }).withPartitioning(.{ .h = .model });

    var out = zml.nn.sdpa(q_sdpa, k_sdpa, v_sdpa, .{ .attn_mask = attn_mask, .allow_cudnn = true });
    out = out.splitAxis(.h, .{ .s = seq_count, .hkv = num_kv_heads, .hg = num_head_groups }).transpose(.{ .s, .q, .hkv, .hg, .hd }).withTags(.{ .s, .b, .hkv, .hg, .hd });

    const out_zero = zml.Tensor.constant(out.dtype().zero()).broad(out.shape());
    out = zml.Tensor.select(valid_q.broad(out.shape()), out, out_zero);
    return out.sum(.s).squeeze(.s).withTags(.{ .b, .hkv, .hg, .hd });
}

fn pagedAttentionVanilla(
    parameters: anytype,
    context: anytype,
    q: zml.Tensor,
    k_cache: zml.Tensor,
    v_cache: zml.Tensor,
    layer_index_: zml.Tensor,
    opts: AttentionOptions,
) zml.Tensor {
    const layer_index = layer_index_.reshape(.{});
    return switch (parameters) {
        .decode => |decode_parameters| pagedAttentionVanillaGroup(
            q,
            k_cache,
            v_cache,
            layer_index,
            decode_parameters.block_table.withPartitioning(.{ .b = .replicated }),
            decode_parameters.cu_seqlens_q.withPartitioning(.{ .b = .replicated }),
            decode_parameters.seqused_k.withPartitioning(.{ .b = .replicated }),
            context.max_seqlen_k,
            opts,
        ),
        .mixed => |mixed_parameters| b: {
            var o = pagedAttentionVanillaGroup(
                q,
                k_cache,
                v_cache,
                layer_index,
                mixed_parameters.block_table_prefill.withPartitioning(.{ .b = .replicated }),
                mixed_parameters.cu_seqlens_q_prefill.withPartitioning(.{ .b = .replicated }),
                mixed_parameters.seqused_k_prefill.withPartitioning(.{ .b = .replicated }),
                context.max_seqlen_k,
                opts,
            );

            const decode_offset = context.decode_offset orelse stdx.debug.panic("Mixed paged attention context requires decode_offset", .{});
            const batch_size_decode = mixed_parameters.block_table_decode.dim(0);
            const q_decode = q.dynamicSlice1d(0, .{ .start = decode_offset, .len = batch_size_decode });
            const o_decode = pagedAttentionVanillaGroup(
                q_decode,
                k_cache,
                v_cache,
                layer_index,
                mixed_parameters.block_table_decode.withPartitioning(.{ .b = .replicated }),
                mixed_parameters.cu_seqlens_q_decode.withPartitioning(.{ .b = .replicated }),
                mixed_parameters.seqused_k_decode.withPartitioning(.{ .b = .replicated }),
                context.max_seqlen_k,
                opts,
            );

            o = o.dynamicUpdateSlice1d(o_decode, 0, decode_offset);
            break :b o;
        },
    };
}

pub fn Wrapper(comptime T: type, run_func: std.meta.DeclEnum(T)) type {
    return struct {
        pub fn register(platform: *const zml.Platform) !void {
            try platform.registerFfi(.{
                .name = T.custom_call_name,
                .platform_name = "cuda",
                .handler = T.run,
                .traits = .{ .command_buffer_compatible = true },
            });
        }

        pub fn run(call_frame: *ffi.CallFrame) callconv(.c) ?*ffi.Error {
            return @field(T, @tagName(run_func))(call_frame) catch b: {
                break :b ffi.Error.create(call_frame.api.?, .unknown, "Unknown");
            };
        }
    };
}

pub const fa2 = struct {
    const custom_call_name = "fa2_mha_varlen_fwd";
    const Wrapped = Wrapper(@This(), .runInner);

    const register = Wrapped.register;
    const run = Wrapped.run;

    pub fn runInner(call_frame: *ffi.CallFrame) !?*ffi.Error {
        if (call_frame.registeringHook()) return null;

        const q = bufferFromFfiBuffer(call_frame.args.buffers()[0]);
        const k = bufferFromFfiBuffer(call_frame.args.buffers()[1]);
        const v = bufferFromFfiBuffer(call_frame.args.buffers()[2]);
        const cu_seqlens_q = bufferFromFfiBuffer(call_frame.args.buffers()[3]);
        const cu_seqlens_k = bufferFromFfiBuffer(call_frame.args.buffers()[4]);
        const softmax_lse = bufferFromFfiBuffer(call_frame.args.buffers()[5]);
        const softmax_lse_accum = bufferFromFfiBuffer(call_frame.args.buffers()[6]);
        const out_accum = bufferFromFfiBuffer(call_frame.args.buffers()[7]);
        const o = bufferFromFfiBuffer(call_frame.results.buffers()[0]);

        const softmax_scale: f32 = getScalarAttributeAs(f32, call_frame, "softmax_scale") orelse b: {
            const head_dim = q.shape.dim(2);
            break :b 1.0 / std.math.sqrt(@as(f32, @floatFromInt(head_dim)));
        };
        const is_causal: bool = getScalarAttributeAs(bool, call_frame, "is_causal").?;
        const window_size_left: i32 = getScalarAttributeAs(i32, call_frame, "window_size_left") orelse -1;
        const window_size_right: i32 = getScalarAttributeAs(i32, call_frame, "window_size_right") orelse -1;
        const max_seqlen_q: i32 = getScalarAttributeAs(i32, call_frame, "max_seqlen_q").?;
        const max_seqlen_k: i32 = getScalarAttributeAs(i32, call_frame, "max_seqlen_k").?;
        const num_heads: i32 = getScalarAttributeAs(i32, call_frame, "num_heads").?;

        const ctx: *ffi.ExecutionContext = @constCast(call_frame.ctx);
        const stream = call_frame.api.stream(ctx);

        const params: flashattn.FA2MhaVarlenFwdParams = .{
            .max_seqlen_q = max_seqlen_q,
            .max_seqlen_k = max_seqlen_k,
            .is_causal = is_causal,
            .softmax_scale = softmax_scale,
            .window_size_left = window_size_left,
            .window_size_right = window_size_right,
            .num_splits = 0,
            .num_heads = num_heads,
        };

        flashattn.fa2_mha_varlen_fwd(
            &q.toFlashattnTensor(),
            &k.toFlashattnTensor(),
            &v.toFlashattnTensor(),
            &o.toFlashattnTensor(),
            &cu_seqlens_q.toFlashattnTensor(),
            &cu_seqlens_k.toFlashattnTensor(),
            null,
            null,
            &softmax_lse.toFlashattnTensor(),
            null,
            &softmax_lse_accum.toFlashattnTensor(),
            &out_accum.toFlashattnTensor(),
            &params,
            stream,
        );

        return null;
    }

    pub const Parameters = struct {
        pub const InitOptions = struct {};

        pub fn init(_: InitOptions) fa2.Parameters {
            return .{};
        }
    };

    pub const Metadata = struct {
        softmax_lse: zml.Tensor,
        softmax_lse_accum: zml.Tensor,
        out_accum: zml.Tensor,

        pub const InitOptions = struct {
            seqlen: i64,
            num_heads: i64,
        };

        pub fn init(opts: InitOptions) Metadata {
            return .{
                .softmax_lse = .fromShape(zml.Shape.init(.{ opts.seqlen, opts.num_heads, 1 }, .f32)
                    .withTags(.{ .s, .h, .dummy })
                    .withPartitioning(.{ .h = .model })),

                .softmax_lse_accum = .fromShape(zml.Shape.init(.{ 1, opts.num_heads, 128 }, .f32)
                    .withTags(.{ .dummy, .h, .hd })
                    .withPartitioning(.{ .h = .model })),

                .out_accum = .fromShape(zml.Shape.init(.{ opts.seqlen, opts.num_heads, 128 }, .f32)
                    .withTags(.{ .s, .h, .hd })
                    .withPartitioning(.{ .h = .model })),
            };
        }

        pub fn initBuffer(self: Metadata, io: std.Io, platform: *const zml.Platform, sharding: zml.sharding.Sharding) !zml.Bufferized(Metadata) {
            return .{
                .softmax_lse = try zml.Buffer.uninitialized(io, platform, self.softmax_lse.shape(), sharding, .{}),
                .softmax_lse_accum = try zml.Buffer.uninitialized(io, platform, self.softmax_lse_accum.shape(), sharding, .{}),
                .out_accum = try zml.Buffer.uninitialized(io, platform, self.out_accum.shape(), sharding, .{}),
            };
        }

        pub fn deinitBuffer(self: *zml.Bufferized(Metadata)) void {
            self.softmax_lse.deinit();
            self.softmax_lse_accum.deinit();
            self.out_accum.deinit();
        }
    };

    pub fn attention(q_: zml.Tensor, k_: zml.Tensor, v_: zml.Tensor, token_index: zml.Tensor, metadata: Metadata, _: Parameters) zml.Tensor {
        const ctx = CompilationContext.current();

        stdx.debug.assert(q_.shape().hasTag(.b) == null or q_.dim(.b) == 1, "fa2.attention support for batch size != 1 is not supported yet.", .{});
        const seqused_k = token_index.addConstant(q_.dim(.q)).reshape(.{1});
        // TODO(Corendos): replace with cumsum
        const cu_seqlens_k = b: {
            const zero = zml.Tensor.constant(token_index.dtype().zero()).reshape(.{1});
            break :b zml.Tensor.concatenate(&.{ zero, seqused_k }, 0).convert(.i32);
        };
        const max_seqlen_q: i32 = @intCast(q_.dim(.q));
        const max_seqlen_k: i32 = @intCast(k_.dim(.k));
        var q, const k, const v = if (q_.shape().hasTag(.b) != null) b: {
            break :b [_]zml.Tensor{
                q_.merge(.{ .tot = .{ .b, .q } }), k_.merge(.{ .tot = .{ .b, .k } }), v_.merge(.{ .tot = .{ .b, .k } }),
            };
        } else b: {
            break :b [_]zml.Tensor{ q_.rename(.{ .q = .tot }), k_.rename(.{ .k = .tot }), v_.rename(.{ .k = .tot }) };
        };
        // TODO(Corendos): replace with cumsum
        const cu_seqlens_q = zml.Tensor.constantTensor(zml.Shape.init(.{2}, .i32), std.mem.sliceAsBytes(&[2]i32{ 0, max_seqlen_q }))
            .withPartitioning(.{ ._0 = .replicated });

        const original_tot = q.dim(.tot);
        const num_heads = q_.dim(.h);
        const num_heads_k = k_.dim(.h);
        const head_size = q_.dim(.hd);
        const ngroups = @divExact(num_heads, num_heads_k);
        const seqlenq_ngroups_swapped = max_seqlen_q == 1 and num_heads > num_heads_k and @mod(head_size, 8) == 0;
        if (seqlenq_ngroups_swapped) {
            q = q.splitAxis(.h, .{ .h = num_heads_k, .ngroups = ngroups }).transpose(.{ .tot, .ngroups, .h, .hd }).merge(.{ .tot = .{ .tot, .ngroups } });
        }

        const q_sharded = q.withPartitioning(.{ .h = .model });
        const model_partitions = ctx.partitioning.numPartitionsForLogicalAxis(q_sharded.shape(), .model) catch unreachable;

        const output_shape = q_sharded.shape();
        var o = zml.ops.manualComputation(
            .{
                q_sharded,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k.withTags(.{.i}).withPartitioning(.{ .i = .replicated }),
                metadata.softmax_lse.withPartitioning(.{ .h = .model }),
                metadata.softmax_lse_accum.withPartitioning(.{ .h = .model }),
                metadata.out_accum.withPartitioning(.{ .h = .model }),
            },
            output_shape,
            .{
                .metadata = .{
                    //.softmax_scale = @as(f32, 1.0),
                    .is_causal = true,
                    .window_size_left = @as(i32, -1),
                    .window_size_right = @as(i32, -1),
                    .max_seqlen_q = max_seqlen_q,
                    .max_seqlen_k = max_seqlen_k,
                    .num_heads = @as(i32, @intCast(@divExact(num_heads, model_partitions))),
                },
                .opts = zml.ops.CustomCallOptions{
                    .output_operand_aliases = &.{0},
                    .has_side_effect = false,
                },
            },
            (struct {
                fn body(ctx_: anytype, _: std.mem.Allocator, sharded_inputs: []const zml.Tensor, output: zml.Shape) zml.Tensor {
                    return zml.ops.customCall(custom_call_name, sharded_inputs, output, ctx_.metadata, ctx_.opts);
                }
            }).body,
        );

        if (seqlenq_ngroups_swapped) {
            o = o.splitAxis(.tot, .{ .tot = original_tot, .ngroups = ngroups }).transpose(.{ .tot, .h, .ngroups, .hd }).merge(.{ .h = .{ .h, .ngroups } });
        }

        return o.splitAxis(.tot, .{ .b = 1, .q = q_.dim(.q) }).squeeze(.b);
    }
};

pub const fa3 = struct {
    const custom_call_name = "fa3_mha_fwd";
    const Wrapped = Wrapper(@This(), .runInner);

    const register = Wrapped.register;
    const run = Wrapped.run;

    pub fn runInner(call_frame: *ffi.CallFrame) !?*ffi.Error {
        if (call_frame.registeringHook()) return null;

        const q = bufferFromFfiBuffer(call_frame.args.buffers()[0]);
        const k = bufferFromFfiBuffer(call_frame.args.buffers()[1]);
        const v = bufferFromFfiBuffer(call_frame.args.buffers()[2]);
        const cu_seqlens_q = bufferFromFfiBuffer(call_frame.args.buffers()[3]);
        const cu_seqlens_k = bufferFromFfiBuffer(call_frame.args.buffers()[4]);
        const softmax_lse = bufferFromFfiBuffer(call_frame.args.buffers()[5]);
        const softmax_lse_accum = bufferFromFfiBuffer(call_frame.args.buffers()[6]);
        const out_accum = bufferFromFfiBuffer(call_frame.args.buffers()[7]);
        const scheduler_metadata = bufferFromFfiBuffer(call_frame.args.buffers()[8]);
        const o = bufferFromFfiBuffer(call_frame.results.buffers()[0]);

        const softmax_scale: f32 = getScalarAttributeAs(f32, call_frame, "softmax_scale") orelse b: {
            const head_dim = q.shape.dim(2);
            break :b 1.0 / std.math.sqrt(@as(f32, @floatFromInt(head_dim)));
        };
        const is_causal: bool = getScalarAttributeAs(bool, call_frame, "is_causal").?;
        const window_size_left: i32 = getScalarAttributeAs(i32, call_frame, "window_size_left") orelse -1;
        const window_size_right: i32 = getScalarAttributeAs(i32, call_frame, "window_size_right") orelse -1;
        const max_seqlen_q: i32 = getScalarAttributeAs(i32, call_frame, "max_seqlen_q").?;
        const max_seqlen_k: i32 = getScalarAttributeAs(i32, call_frame, "max_seqlen_k").?;

        const ctx: *ffi.ExecutionContext = @constCast(call_frame.ctx);
        const stream = call_frame.api.stream(ctx);

        const params: flashattn.FA3MhaFwdParams = .{
            .max_seqlen_q = max_seqlen_q,
            .max_seqlen_k = max_seqlen_k,
            .softcap = 0.0,
            .is_rotary_interleaved = false,
            .num_splits = 0,
            .sm_margin = 0,
            .is_causal = is_causal,
            .softmax_scale = softmax_scale,
            .window_size_left = window_size_left,
            .window_size_right = window_size_right,
            .cp_world_size = 1,
            .cp_rank = 0,
        };

        flashattn.fa3_mha_fwd(
            &q.toFlashattnTensor(),
            &k.toFlashattnTensor(),
            &v.toFlashattnTensor(),
            &o.toFlashattnTensor(),
            &cu_seqlens_q.toFlashattnTensor(),
            &cu_seqlens_k.toFlashattnTensor(),
            null,
            null,
            null,
            null,
            null,
            null,
            &softmax_lse.toFlashattnTensor(),
            &softmax_lse_accum.toFlashattnTensor(),
            &out_accum.toFlashattnTensor(),
            &scheduler_metadata.toFlashattnTensor(),
            null,
            null,
            &params,
            stream,
        );

        return null;
    }

    pub const Parameters = struct {
        pub const InitOptions = struct {};

        pub fn init(opts: InitOptions) fa3.Parameters {
            _ = opts;
            return .{};
        }
    };

    pub const Metadata = struct {
        softmax_lse: zml.Tensor,
        softmax_lse_accum: zml.Tensor,
        out_accum: zml.Tensor,
        scheduler_metadata: zml.Tensor,

        pub const InitOptions = struct {
            seqlen: i64,
            num_heads: i64,
        };

        pub fn init(opts: InitOptions) Metadata {
            return .{
                .softmax_lse = .fromShape(zml.Shape.init(.{opts.num_heads * opts.seqlen * 4}, .i8)
                    .withTags(.{.h}).withPartitioning(.{ .h = .model })),
                .softmax_lse_accum = .fromShape(zml.Shape.init(.{opts.num_heads * 128 * 4}, .i8)
                    .withTags(.{.h}).withPartitioning(.{ .h = .model })),
                .out_accum = .fromShape(zml.Shape.init(.{opts.num_heads * opts.seqlen * 128 * 4}, .i8)
                    .withTags(.{.h}).withPartitioning(.{ .h = .model })),
                .scheduler_metadata = .fromShape(zml.Shape.init(.{2}, .i32)
                    .withTags(.{.meta}).withPartitioning(.{ .meta = .replicated })),
            };
        }

        pub fn initBuffer(self: Metadata, io: std.Io, platform: *const zml.Platform, sharding: zml.sharding.Sharding) !zml.Bufferized(Metadata) {
            return .{
                .softmax_lse = try zml.Buffer.uninitialized(io, platform, self.softmax_lse.shape(), sharding, .{}),
                .softmax_lse_accum = try zml.Buffer.uninitialized(io, platform, self.softmax_lse_accum.shape(), sharding, .{}),
                .out_accum = try zml.Buffer.uninitialized(io, platform, self.out_accum.shape(), sharding, .{}),
                .scheduler_metadata = try zml.Buffer.uninitialized(io, platform, self.scheduler_metadata.shape(), sharding, .{}),
            };
        }

        pub fn deinitBuffer(self: *zml.Bufferized(Metadata)) void {
            self.softmax_lse.deinit();
            self.softmax_lse_accum.deinit();
            self.out_accum.deinit();
            self.scheduler_metadata.deinit();
        }
    };

    pub fn attention(q_: zml.Tensor, k_: zml.Tensor, v_: zml.Tensor, token_index: zml.Tensor, metadata: Metadata, _: Parameters) zml.Tensor {
        stdx.debug.assert(q_.shape().hasTag(.b) == null or q_.dim(.b) == 1, "fa3.attention support for batch size != 1 is not supported yet.", .{});
        const seqused_k = token_index.addConstant(q_.dim(.q)).reshape(.{1});
        // TODO(Corendos): replace with cumsum
        const cu_seqlens_k = b: {
            const zero = zml.Tensor.constant(token_index.dtype().zero()).reshape(.{1});
            break :b zml.Tensor.concatenate(&.{ zero, seqused_k }, 0).convert(.i32);
        };
        const max_seqlen_q: i32 = @intCast(q_.dim(.q));
        const max_seqlen_k: i32 = @intCast(k_.dim(.k));
        var q = q_.insertAxes(.q, .{.b}).merge(.{ .tot = .{ .b, .q } });
        const k = k_.insertAxes(.k, .{.b}).merge(.{ .tot = .{ .b, .k } });
        const v = v_.insertAxes(.k, .{.b}).merge(.{ .tot = .{ .b, .k } });
        // TODO(Corendos): replace with cumsum
        const cu_seqlens_q = zml.Tensor.constantTensor(zml.Shape.init(.{2}, .i32), std.mem.sliceAsBytes(&[2]i32{ 0, max_seqlen_q }))
            .withPartitioning(.{ ._0 = .replicated });

        var o = zml.ops.customCall(
            custom_call_name,
            .{
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                metadata.softmax_lse,
                metadata.softmax_lse_accum,
                metadata.out_accum,
                metadata.scheduler_metadata,
            },
            .{q.shape()},
            .{
                .is_causal = true,
                .window_size_left = @as(i32, -1),
                .window_size_right = @as(i32, -1),
                .max_seqlen_q = max_seqlen_q,
                .max_seqlen_k = max_seqlen_k,
            },
            .{
                .output_operand_aliases = &.{0},
                .has_side_effect = false,
            },
        );

        return o.splitAxis(.tot, .{ .b = 1, .q = q_.dim(.q) }).squeeze(.b);
    }
};

pub const paged_fa2 = struct {
    // God knows why flash attention uses this number and not something else.
    const MAX_NUM_SPLITS = 8;

    pub const Variant = enum {
        decode,
        mixed,
    };

    pub const Options = union(Variant) {
        decode: DecodeOptions,
        mixed: MixedOptions,

        pub fn isPrefill(self: Options) bool {
            return self == .mixed;
        }

        pub fn maxNumPages(self: Options) usize {
            return switch (self) {
                .decode => |decode_options| decode_options.max_num_pages,
                .mixed => |mixed_options| mixed_options.max_num_pages,
            };
        }
    };

    pub const Parameters = union(Variant) {
        decode: DecodeParameters,
        mixed: MixedParameters,

        pub fn init(options_: Options) Parameters {
            return switch (options_) {
                .decode => |decode_options| .{ .decode = DecodeParameters.init(decode_options) },
                .mixed => |mixed_options| .{ .mixed = MixedParameters.init(mixed_options) },
            };
        }

        pub fn allocationSize(self: Parameters) usize {
            return switch (self) {
                .decode => |decode| decode.allocationSize(),
                .mixed => |mixed| mixed.allocationSize(),
            };
        }

        pub fn options(self: Parameters) Options {
            return switch (self) {
                .decode => |v| .{ .decode = v.options },
                .mixed => |v| .{ .mixed = v.options },
            };
        }
    };

    pub const DecodeOptions = struct {
        batch_size: usize,
        max_num_pages: usize,
        max_seqlen_k: usize,
        max_token_count: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    };

    pub const DecodeParameters = struct {
        block_table: zml.Tensor,
        cu_seqlens_q: zml.Tensor,
        seqused_k: zml.Tensor,

        metadata: DecodeMetadata,
        options: DecodeOptions,

        pub fn init(options: DecodeOptions) DecodeParameters {
            return .{
                .block_table = zml.Tensor.init(.{ .b = options.batch_size, .p = options.max_num_pages }, .i32),
                .cu_seqlens_q = zml.Tensor.init(.{ .b = options.batch_size + 1 }, .i32),
                .seqused_k = zml.Tensor.init(.{ .b = options.batch_size }, .i32),
                .metadata = DecodeMetadata.init(options),
                .options = options,
            };
        }

        pub fn allocationSize(self: DecodeParameters) usize {
            var allocation_size: usize = 0;

            allocation_size += self.block_table.byteSize();
            allocation_size += self.cu_seqlens_q.byteSize();
            allocation_size += self.seqused_k.byteSize();
            allocation_size += self.metadata.allocationSize();

            return allocation_size;
        }
    };

    pub const DecodeMetadata = struct {
        out_accum: zml.Tensor,

        pub fn init(opts: DecodeOptions) DecodeMetadata {
            const shape = zml.Shape.init(.{
                .splits = MAX_NUM_SPLITS,
                .hkv = opts.num_kv_heads,
                .hg = @divExact(opts.num_heads, opts.num_kv_heads),
                .b = opts.max_token_count,
                .hd = opts.head_dim,
            }, .f32).withPartitioning(.{ .hkv = .model });
            return .{
                .out_accum = .fromShape(shape),
            };
        }

        pub fn initBuffers(self: *const DecodeMetadata, io: std.Io, platform: zml.Platform, out_accum_sharding: zml.sharding.Sharding) !zml.Bufferized(DecodeMetadata) {
            return .{
                .out_accum = try zml.Buffer.uninitialized(io, &platform, self.out_accum.shape(), out_accum_sharding, .{}),
            };
        }

        pub fn deinitBuffers(self: *const zml.Bufferized(DecodeMetadata)) void {
            self.out_accum.deinit();
        }

        pub fn allocationSize(self: DecodeMetadata) usize {
            var allocation_size: usize = 0;
            allocation_size += self.out_accum.byteSize();
            return allocation_size;
        }
    };

    pub const MixedOptions = struct {
        batch_size_prefill: usize,
        batch_size_decode: usize,
        max_num_pages: usize,
        max_seqlen_k: usize,
        max_token_count: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    };

    pub const MixedParameters = struct {
        block_table_prefill: zml.Tensor,
        cu_seqlens_q_prefill: zml.Tensor,
        seqused_k_prefill: zml.Tensor,

        block_table_decode: zml.Tensor,
        cu_seqlens_q_decode: zml.Tensor,
        seqused_k_decode: zml.Tensor,

        metadata: MixedMetadata,
        options: MixedOptions,

        pub fn init(options: MixedOptions) MixedParameters {
            return .{
                .block_table_prefill = zml.Tensor.init(
                    .{ .b = options.batch_size_prefill, .p = options.max_num_pages },
                    .i32,
                ),
                .cu_seqlens_q_prefill = zml.Tensor.init(.{ .b = options.batch_size_prefill + 1 }, .i32),
                .seqused_k_prefill = zml.Tensor.init(.{ .b = options.batch_size_prefill }, .i32),
                .block_table_decode = zml.Tensor.init(.{ .b = options.batch_size_decode, .p = options.max_num_pages }, .i32),
                .cu_seqlens_q_decode = zml.Tensor.init(.{ .b = options.batch_size_decode + 1 }, .i32),
                .seqused_k_decode = zml.Tensor.init(.{ .b = options.batch_size_decode }, .i32),
                .metadata = MixedMetadata.init(options),
                .options = options,
            };
        }

        pub fn allocationSize(self: MixedParameters) usize {
            var allocation_size: usize = 0;
            allocation_size += self.block_table_decode.byteSize();
            allocation_size += self.cu_seqlens_q_prefill.byteSize();
            allocation_size += self.seqused_k_prefill.byteSize();

            allocation_size += self.block_table_decode.byteSize();
            allocation_size += self.cu_seqlens_q_decode.byteSize();
            allocation_size += self.seqused_k_decode.byteSize();
            allocation_size += self.metadata.allocationSize();

            return allocation_size;
        }
    };

    pub const MixedMetadata = struct {
        out_accum: zml.Tensor,
        host_metadata: zml.Tensor,

        pub fn init(opts: MixedOptions) MixedMetadata {
            const shape = zml.Shape.init(.{
                .splits = MAX_NUM_SPLITS,
                .hkv = opts.num_kv_heads,
                .hg = @divExact(opts.num_heads, opts.num_kv_heads),
                .b = opts.max_token_count,
                .hd = opts.head_dim,
            }, .f32).withPartitioning(.{ .hkv = .model });
            return .{
                .out_accum = .fromShape(shape),
                .host_metadata = .init(.{2}, .i32),
            };
        }

        pub fn initBuffers(self: *const MixedMetadata, io: std.Io, platform: zml.Platform, out_accum_sharding: zml.sharding.Sharding) !zml.Bufferized(MixedMetadata) {
            return .{
                .out_accum = try zml.Buffer.uninitialized(io, &platform, self.out_accum.shape(), out_accum_sharding, .{}),
                .host_metadata = undefined,
            };
        }

        pub fn deinitBuffers(self: *const zml.Bufferized(MixedMetadata)) void {
            self.out_accum.deinit();
        }

        pub fn allocationSize(self: MixedMetadata) usize {
            var allocation_size: usize = 0;
            allocation_size += self.out_accum.byteSize();
            allocation_size += self.host_metadata.byteSize();
            return allocation_size;
        }
    };

    pub const Context = struct {
        max_seqlen_k: usize,
        decode_offset: ?zml.Tensor = null,

        pub fn init(parameters: Parameters, num_heads: i64, num_kv_heads: i64, head_dim: i64, page_size: i64) Context {
            _ = num_heads;
            _ = num_kv_heads;
            _ = head_dim;
            _ = page_size;
            const max_seqlen_k = switch (parameters) {
                inline else => |v| v.options.max_seqlen_k,
            };
            const decode_offset = switch (parameters) {
                .mixed => |mixed_parameters| mixed_parameters.metadata.host_metadata.slice1d(0, .{ .end = 1 }).squeeze(0),
                .decode => null,
            };
            return .{ .max_seqlen_k = max_seqlen_k, .decode_offset = decode_offset };
        }
    };

    pub const Prefill = struct {
        pub const custom_call_name = "paged_fa2_prefill";
        const Wrapped = Wrapper(@This(), .runInner);

        const register = Wrapped.register;
        const run = Wrapped.run;

        pub fn runInner(call_frame: *ffi.CallFrame) !?*ffi.Error {
            if (call_frame.registeringHook()) return null;

            const layer_index = bufferFromFfiBuffer(call_frame.args.buffers()[11]);
            const layer_index_raw = @as([*]i32, @ptrCast(@alignCast(layer_index.ptr)))[0];

            const q = bufferFromFfiBuffer(call_frame.args.buffers()[0]);
            const paged_k = fixupKvCacheBuffer(bufferFromFfiBuffer(call_frame.args.buffers()[1]), layer_index_raw);
            const paged_v = fixupKvCacheBuffer(bufferFromFfiBuffer(call_frame.args.buffers()[2]), layer_index_raw);
            const cu_seqlens_q = bufferFromFfiBuffer(call_frame.args.buffers()[3]);
            const cu_seqlens_k = bufferFromFfiBuffer(call_frame.args.buffers()[4]);
            const seqused_k = bufferFromFfiBuffer(call_frame.args.buffers()[5]);
            const block_table = bufferFromFfiBuffer(call_frame.args.buffers()[6]);
            const softmax_lse = bufferFromFfiBuffer(call_frame.args.buffers()[7]);
            const softmax_lse_accum = bufferFromFfiBuffer(call_frame.args.buffers()[8]);
            const out_accum = bufferFromFfiBuffer(call_frame.args.buffers()[9]);
            const host_metadata = bufferFromFfiBuffer(call_frame.args.buffers()[10]);
            const o = bufferFromFfiBuffer(call_frame.results.buffers()[0]);

            const max_seqlen_q = @as([*]i32, @ptrCast(@alignCast(host_metadata.ptr)))[1];
            const softmax_scale: f32 = getScalarAttributeAs(f32, call_frame, "softmax_scale") orelse b: {
                const head_dim = q.shape.dim(2);
                break :b 1.0 / std.math.sqrt(@as(f32, @floatFromInt(head_dim)));
            };
            const is_causal: bool = getScalarAttributeAs(bool, call_frame, "is_causal").?;
            const window_size_left: i32 = getScalarAttributeAs(i32, call_frame, "window_size_left") orelse -1;
            const window_size_right: i32 = getScalarAttributeAs(i32, call_frame, "window_size_right") orelse -1;
            const max_seqlen_k: i32 = getScalarAttributeAs(i32, call_frame, "max_seqlen_k").?;
            const num_heads: i32 = getScalarAttributeAs(i32, call_frame, "num_heads").?;

            const ctx: *ffi.ExecutionContext = @constCast(call_frame.ctx);
            const stream = call_frame.api.stream(ctx);

            const params: flashattn.FA2MhaVarlenFwdParams = .{
                .max_seqlen_q = max_seqlen_q,
                .max_seqlen_k = max_seqlen_k,
                .is_causal = is_causal,
                .softmax_scale = softmax_scale,
                .window_size_left = window_size_left,
                .window_size_right = window_size_right,
                .num_splits = MAX_NUM_SPLITS,
                .num_heads = num_heads,
            };

            flashattn.fa2_mha_varlen_fwd(
                &q.toFlashattnTensor(),
                &paged_k.toFlashattnTensor(),
                &paged_v.toFlashattnTensor(),
                &o.toFlashattnTensor(),
                &cu_seqlens_q.toFlashattnTensor(),
                &cu_seqlens_k.toFlashattnTensor(),
                &seqused_k.toFlashattnTensor(),
                &block_table.toFlashattnTensor(),
                &softmax_lse.toFlashattnTensor(),
                null,
                &softmax_lse_accum.toFlashattnTensor(),
                &out_accum.toFlashattnTensor(),
                &params,
                stream,
            );

            return null;
        }
    };

    pub const Decode = struct {
        pub const custom_call_name = "paged_fa2_decode";
        const Wrapped = Wrapper(@This(), .runInner);

        const register = Wrapped.register;
        const run = Wrapped.run;

        pub fn runInner(call_frame: *ffi.CallFrame) !?*ffi.Error {
            if (call_frame.registeringHook()) return null;

            const layer_index = bufferFromFfiBuffer(call_frame.args.buffers()[10]);
            const layer_index_raw = @as([*]i32, @ptrCast(@alignCast(layer_index.ptr)))[0];

            const q = bufferFromFfiBuffer(call_frame.args.buffers()[0]);
            const paged_k = fixupKvCacheBuffer(bufferFromFfiBuffer(call_frame.args.buffers()[1]), layer_index_raw);
            const paged_v = fixupKvCacheBuffer(bufferFromFfiBuffer(call_frame.args.buffers()[2]), layer_index_raw);
            const cu_seqlens_q = bufferFromFfiBuffer(call_frame.args.buffers()[3]);
            const cu_seqlens_k = bufferFromFfiBuffer(call_frame.args.buffers()[4]);
            const seqused_k = bufferFromFfiBuffer(call_frame.args.buffers()[5]);
            const block_table = bufferFromFfiBuffer(call_frame.args.buffers()[6]);
            const softmax_lse = bufferFromFfiBuffer(call_frame.args.buffers()[7]);
            const softmax_lse_accum = bufferFromFfiBuffer(call_frame.args.buffers()[8]);
            const out_accum = bufferFromFfiBuffer(call_frame.args.buffers()[9]);
            const o = bufferFromFfiBuffer(call_frame.results.buffers()[0]);

            const softmax_scale: f32 = getScalarAttributeAs(f32, call_frame, "softmax_scale") orelse b: {
                const head_dim = q.shape.dim(2);
                break :b 1.0 / std.math.sqrt(@as(f32, @floatFromInt(head_dim)));
            };
            const is_causal: bool = getScalarAttributeAs(bool, call_frame, "is_causal").?;
            const window_size_left: i32 = getScalarAttributeAs(i32, call_frame, "window_size_left") orelse -1;
            const window_size_right: i32 = getScalarAttributeAs(i32, call_frame, "window_size_right") orelse -1;
            const max_seqlen_k: i32 = getScalarAttributeAs(i32, call_frame, "max_seqlen_k").?;
            const num_heads: i32 = getScalarAttributeAs(i32, call_frame, "num_heads").?;

            const ctx: *ffi.ExecutionContext = @constCast(call_frame.ctx);
            const stream = call_frame.api.stream(ctx);

            const params: flashattn.FA2MhaVarlenFwdParams = .{
                .max_seqlen_q = 1,
                .max_seqlen_k = max_seqlen_k,
                .is_causal = is_causal,
                .softmax_scale = softmax_scale,
                .window_size_left = window_size_left,
                .window_size_right = window_size_right,
                .num_splits = MAX_NUM_SPLITS,
                .num_heads = num_heads,
            };

            flashattn.fa2_mha_varlen_fwd(
                &q.toFlashattnTensor(),
                &paged_k.toFlashattnTensor(),
                &paged_v.toFlashattnTensor(),
                &o.toFlashattnTensor(),
                &cu_seqlens_q.toFlashattnTensor(),
                &cu_seqlens_k.toFlashattnTensor(),
                &seqused_k.toFlashattnTensor(),
                &block_table.toFlashattnTensor(),
                &softmax_lse.toFlashattnTensor(),
                null,
                &softmax_lse_accum.toFlashattnTensor(),
                &out_accum.toFlashattnTensor(),
                &params,
                stream,
            );

            return null;
        }
    };

    pub fn pagedAttention(parameters: Parameters, context: Context, q: zml.Tensor, k_cache: zml.Tensor, v_cache: zml.Tensor, layer_index_: zml.Tensor, opts: AttentionOptions) zml.Tensor {
        stdx.debug.assert(q.shape().hasTags(.{ .b, .hg, .hkv, .hd }), "Expected q to have tags .b, .h, .hd", .{});
        stdx.debug.assert(k_cache.shape().hasTags(.{ .page, .k_chunk, .hkv, .hd }), "Expected paged_k to have tags .page, .k_chunk, .h, .hd, got {}", .{k_cache.shape()});
        stdx.debug.assert(v_cache.shape().hasTags(.{ .page, .k_chunk, .hkv, .hd }), "Expected paged_v to have tags .page, .k_chunk, .h, .hd. got {}", .{v_cache.shape()});
        if (q.dim(.hd) > 256) {
            return pagedAttentionVanilla(parameters, context, q, k_cache, v_cache, layer_index_, opts);
        }
        const ctx = CompilationContext.current();

        const num_head_groups = q.dim(.hg);
        const num_kv_heads = q.dim(.hkv);
        const head_dim = q.dim(.hd);
        const num_heads = num_head_groups * num_kv_heads;
        const layer_index = layer_index_.reshape(.{1}).withPartitioning(.{ ._0 = .replicated });
        // FIXME: remove unreachable and propagate error correctly.
        const num_heads_per_shard = @divExact(num_heads, ctx.partitioning.numPartitionsForLogicalAxis(q.shape(), .model) catch unreachable);

        const o = switch (parameters) {
            .decode => |decode_parameters| b: {
                const seqlenq_ngroups_swapped = num_heads > num_kv_heads and @mod(head_dim, 8) == 0 and opts.sliding_window < 0;

                const block_table = decode_parameters.block_table.withPartitioning(.{ .b = .replicated });
                const cu_seqlens_q = decode_parameters.cu_seqlens_q.withPartitioning(.{ .b = .replicated });
                const seqused_k = decode_parameters.seqused_k.withPartitioning(.{ .b = .replicated });

                const out_accum = zml.Tensor.constant(zml.DataType.f32.zero()).broad(.init(.{
                    .splits = MAX_NUM_SPLITS,
                    .hkv = num_kv_heads,
                    .hg = num_head_groups,
                    .b = q.dim(.b),
                    .hd = head_dim,
                }, .f32)).withPartitioning(.{ .hkv = .model });
                const softmax_lse = zml.Tensor.constant(zml.DataType.f32.zero()).broad(.init(.{
                    .hkv = num_kv_heads,
                    .hg = num_head_groups,
                    .b = q.dim(.b),
                }, .f32)).withPartitioning(.{ .hkv = .model });
                const softmax_lse_accum = zml.Tensor.constant(zml.DataType.f32.zero()).broad(.init(.{
                    .splits = MAX_NUM_SPLITS,
                    .hkv = num_kv_heads,
                    .hg = num_head_groups,
                    .b = q.dim(.b),
                }, .f32)).withPartitioning(.{ .hkv = .model });
                const dummy_cu_seqlens_k = zml.Tensor.constant(zml.DataType.i32.zero()).broad(cu_seqlens_q.shape());

                const batch_dim = q.dim(.b);
                var q2 = q;
                if (seqlenq_ngroups_swapped) {
                    q2 = q2.transpose(.{ .b, .hg, .hkv, .hd }).merge(.{ .b = .{ .b, .hg } }).withPartitioning(.{ .hkv = .model });
                } else {
                    q2 = q2.transpose(.{ .b, .hkv, .hg, .hd }).merge(.{ .h = .{ .hkv, .hg } }).withPartitioning(.{ .h = .model });
                }

                const output_shape = q2.shape();
                var o = zml.ops.manualComputation(
                    .{
                        q2,
                        k_cache,
                        v_cache,
                        cu_seqlens_q,
                        dummy_cu_seqlens_k,
                        seqused_k,
                        block_table,
                        softmax_lse,
                        softmax_lse_accum,
                        out_accum,
                        layer_index,
                    },
                    output_shape,
                    .{
                        .metadata = .{
                            .is_causal = opts.is_causal,
                            .max_seqlen_k = context.max_seqlen_k,
                            .num_heads = num_heads_per_shard,
                            .window_size_left = opts.sliding_window,
                        },
                        .opts = zml.ops.CustomCallOptions{
                            .has_side_effect = false,
                        },
                    },
                    (struct {
                        fn body(ctx_: anytype, _: std.mem.Allocator, sharded_inputs: []const zml.Tensor, output: zml.Shape) zml.Tensor {
                            return zml.ops.customCall(Decode.custom_call_name, sharded_inputs, output, ctx_.metadata, ctx_.opts);
                        }
                    }).body,
                );

                if (seqlenq_ngroups_swapped) {
                    o = o.splitAxis(.b, .{ .b = batch_dim, .hg = num_head_groups }).transpose(.{ .b, .hkv, .hg, .hd });
                } else {
                    o = o.splitAxis(.h, .{ .hkv = num_kv_heads, .hg = num_head_groups });
                }

                break :b o;
            },
            .mixed => |mixed_parameters| b: {
                const seqlenq_ngroups_swapped = num_heads > num_kv_heads and @mod(head_dim, 8) == 0 and opts.sliding_window < 0;

                const block_table_prefill = mixed_parameters.block_table_prefill.withPartitioning(.{ .b = .replicated });
                const cu_seqlens_q_prefill = mixed_parameters.cu_seqlens_q_prefill.withPartitioning(.{ .b = .replicated });
                const seqused_k_prefill = mixed_parameters.seqused_k_prefill.withPartitioning(.{ .b = .replicated });
                const block_table_decode = mixed_parameters.block_table_decode.withPartitioning(.{ .b = .replicated });
                const cu_seqlens_q_decode = mixed_parameters.cu_seqlens_q_decode.withPartitioning(.{ .b = .replicated });
                const seqused_k_decode = mixed_parameters.seqused_k_decode.withPartitioning(.{ .b = .replicated });
                const host_metadata = mixed_parameters.metadata.host_metadata.withTags(.{.x}).withPartitioning(.{ .x = .replicated });

                const out_accum_prefill = zml.Tensor.constant(zml.DataType.f32.zero()).broad(.init(.{
                    .splits = MAX_NUM_SPLITS,
                    .hkv = num_kv_heads,
                    .hg = num_head_groups,
                    .b = q.dim(.b),
                    .hd = head_dim,
                }, .f32)).withPartitioning(.{ .hkv = .model });
                const softmax_lse_prefill = zml.Tensor.constant(zml.DataType.f32.zero()).broad(.init(.{
                    .hkv = num_kv_heads,
                    .hg = num_head_groups,
                    .b = q.dim(.b),
                }, .f32)).withPartitioning(.{ .hkv = .model });
                const softmax_lse_accum_prefill = zml.Tensor.constant(zml.DataType.f32.zero()).broad(.init(.{
                    .splits = MAX_NUM_SPLITS,
                    .hkv = num_kv_heads,
                    .hg = num_head_groups,
                    .b = q.dim(.b),
                }, .f32)).withPartitioning(.{ .hkv = .model });
                const dummy_cu_seqlens_k_prefill = zml.Tensor.constant(zml.DataType.i32.zero()).broad(cu_seqlens_q_prefill.shape());

                var q2 = q;
                q2 = q2.transpose(.{ .b, .hkv, .hg, .hd }).merge(.{ .h = .{ .hkv, .hg } }).withPartitioning(.{ .h = .model });

                const output_shape = q2.shape();
                var o = zml.ops.manualComputation(
                    .{
                        q2,
                        k_cache,
                        v_cache,
                        cu_seqlens_q_prefill,
                        dummy_cu_seqlens_k_prefill,
                        seqused_k_prefill,
                        block_table_prefill,
                        softmax_lse_prefill,
                        softmax_lse_accum_prefill,
                        out_accum_prefill,
                        host_metadata,
                        layer_index,
                    },
                    output_shape,
                    .{
                        .metadata = .{
                            .is_causal = opts.is_causal,
                            .max_seqlen_k = context.max_seqlen_k,
                            .num_heads = num_heads_per_shard,
                            .window_size_left = opts.sliding_window,
                        },
                        .opts = zml.ops.CustomCallOptions{
                            .has_side_effect = false,
                        },
                    },
                    (struct {
                        fn body(ctx_: anytype, _: std.mem.Allocator, sharded_inputs: []const zml.Tensor, output: zml.Shape) zml.Tensor {
                            return zml.ops.customCall(Prefill.custom_call_name, sharded_inputs, output, ctx_.metadata, ctx_.opts);
                        }
                    }).body,
                );

                o = o.splitAxis(.h, .{ .hkv = num_kv_heads, .hg = num_head_groups });

                const batch_dim_decode = block_table_decode.dim(0);
                const out_accum_decode = zml.Tensor.constant(zml.DataType.f32.zero()).broad(.init(.{
                    .splits = MAX_NUM_SPLITS,
                    .hkv = num_kv_heads,
                    .hg = num_head_groups,
                    .b = batch_dim_decode,
                    .hd = head_dim,
                }, .f32)).withPartitioning(.{ .hkv = .model });
                const softmax_lse_decode = zml.Tensor.constant(zml.DataType.f32.zero()).broad(.init(.{
                    .hkv = num_kv_heads,
                    .hg = num_head_groups,
                    .b = batch_dim_decode,
                }, .f32)).withPartitioning(.{ .hkv = .model });
                const softmax_lse_accum_decode = zml.Tensor.constant(zml.DataType.f32.zero()).broad(.init(.{
                    .splits = MAX_NUM_SPLITS,
                    .hkv = num_kv_heads,
                    .hg = num_head_groups,
                    .b = batch_dim_decode,
                }, .f32)).withPartitioning(.{ .hkv = .model });
                const dummy_cu_seqlens_k_decode = zml.Tensor.constant(zml.DataType.i32.zero()).broad(cu_seqlens_q_decode.shape());
                var q_decode = q.dynamicSlice1d(0, .{ .start = context.decode_offset.?, .len = batch_dim_decode });

                if (seqlenq_ngroups_swapped) {
                    q_decode = q_decode.transpose(.{ .b, .hg, .hkv, .hd }).merge(.{ .b = .{ .b, .hg } }).withPartitioning(.{ .hkv = .model });
                } else {
                    q_decode = q_decode.transpose(.{ .b, .hkv, .hg, .hd }).merge(.{ .h = .{ .hkv, .hg } }).withPartitioning(.{ .h = .model });
                }

                const output_shape_decode = q_decode.shape();
                var o_decode = zml.ops.manualComputation(
                    .{
                        q_decode,
                        k_cache,
                        v_cache,
                        cu_seqlens_q_decode,
                        dummy_cu_seqlens_k_decode,
                        seqused_k_decode,
                        block_table_decode,
                        softmax_lse_decode,
                        softmax_lse_accum_decode,
                        out_accum_decode,
                        layer_index,
                    },
                    output_shape_decode,
                    .{
                        .metadata = .{
                            .is_causal = opts.is_causal,
                            .max_seqlen_k = context.max_seqlen_k,
                            .num_heads = num_heads_per_shard,
                            .window_size_left = opts.sliding_window,
                        },
                        .opts = zml.ops.CustomCallOptions{
                            .has_side_effect = false,
                        },
                    },
                    (struct {
                        fn body(ctx_: anytype, _: std.mem.Allocator, sharded_inputs: []const zml.Tensor, output: zml.Shape) zml.Tensor {
                            return zml.ops.customCall(Decode.custom_call_name, sharded_inputs, output, ctx_.metadata, ctx_.opts);
                        }
                    }).body,
                );

                if (seqlenq_ngroups_swapped) {
                    o_decode = o_decode.splitAxis(.b, .{ .b = batch_dim_decode, .hg = num_head_groups }).transpose(.{ .b, .hkv, .hg, .hd });
                } else {
                    o_decode = o_decode.splitAxis(.h, .{ .hkv = num_kv_heads, .hg = num_head_groups });
                }

                o = o.dynamicUpdateSlice1d(o_decode, 0, context.decode_offset.?);
                break :b o;
            },
        };

        return o;
    }
};

pub const paged_fa3 = struct {
    // God knows why flash attention uses this number and not something else.
    const MAX_NUM_SPLITS = 32;

    pub const Variant = enum {
        decode,
        mixed,
    };

    pub const Options = union(Variant) {
        decode: DecodeOptions,
        mixed: MixedOptions,

        pub fn isPrefill(self: Options) bool {
            return self == .mixed;
        }

        pub fn maxNumPages(self: Options) usize {
            return switch (self) {
                .decode => |decode_options| decode_options.max_num_pages,
                .mixed => |mixed_options| mixed_options.max_num_pages,
            };
        }
    };

    pub const Parameters = union(Variant) {
        decode: DecodeParameters,
        mixed: MixedParameters,

        pub fn init(options_: Options) Parameters {
            return switch (options_) {
                .decode => |decode_options| .{ .decode = DecodeParameters.init(decode_options) },
                .mixed => |mixed_options| .{ .mixed = MixedParameters.init(mixed_options) },
            };
        }

        pub fn allocationSize(self: Parameters) usize {
            return switch (self) {
                .decode => |decode| decode.allocationSize(),
                .mixed => |mixed| mixed.allocationSize(),
            };
        }

        pub fn options(self: Parameters) Options {
            return switch (self) {
                .decode => |v| .{ .decode = v.options },
                .mixed => |v| .{ .mixed = v.options },
            };
        }
    };

    pub const DecodeOptions = struct {
        batch_size: usize,
        max_num_pages: usize,
        max_seqlen_k: usize,
        max_token_count: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    };

    pub const DecodeParameters = struct {
        block_table: zml.Tensor,
        cu_seqlens_q: zml.Tensor,
        seqused_k: zml.Tensor,

        metadata: DecodeMetadata,
        options: DecodeOptions,

        pub fn init(options: DecodeOptions) DecodeParameters {
            return .{
                .block_table = .init(.{ .b = options.batch_size, .p = options.max_num_pages }, .i32),
                .cu_seqlens_q = .init(.{ .b = options.batch_size + 1 }, .i32),
                .seqused_k = .init(.{ .b = options.batch_size }, .i32),
                .metadata = DecodeMetadata.init(options),
                .options = options,
            };
        }

        pub fn allocationSize(self: DecodeParameters) usize {
            var allocation_size: usize = 0;

            allocation_size += self.block_table.byteSize();
            allocation_size += self.cu_seqlens_q.byteSize();
            allocation_size += self.seqused_k.byteSize();
            allocation_size += self.metadata.allocationSize();

            return allocation_size;
        }
    };

    pub const DecodeMetadata = struct {
        out_accum: zml.Tensor,

        pub fn init(opts: DecodeOptions) DecodeMetadata {
            const shape = zml.Shape.init(.{
                .splits = MAX_NUM_SPLITS,
                .hkv = opts.num_kv_heads,
                .hg = @divExact(opts.num_heads, opts.num_kv_heads),
                .b = opts.max_token_count,
                .hd = opts.head_dim,
            }, .f32).withPartitioning(.{ .hkv = .model });
            return .{
                .out_accum = .fromShape(shape),
            };
        }

        pub fn initBuffers(self: *const DecodeMetadata, io: std.Io, platform: zml.Platform, out_accum_sharding: zml.sharding.Sharding) !zml.Bufferized(DecodeMetadata) {
            return .{
                .out_accum = try zml.Buffer.uninitialized(io, &platform, self.out_accum.shape(), out_accum_sharding, .{}),
            };
        }

        pub fn deinitBuffers(self: *const zml.Bufferized(DecodeMetadata)) void {
            self.out_accum.deinit();
        }

        pub fn allocationSize(self: DecodeMetadata) usize {
            var allocation_size: usize = 0;
            allocation_size += self.out_accum.byteSize();
            return allocation_size;
        }
    };

    pub const MixedOptions = struct {
        batch_size_prefill: usize,
        batch_size_decode: usize,
        max_num_pages: usize,
        max_seqlen_k: usize,
        max_token_count: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    };

    pub const MixedParameters = struct {
        block_table_prefill: zml.Tensor,
        cu_seqlens_q_prefill: zml.Tensor,
        seqused_k_prefill: zml.Tensor,

        block_table_decode: zml.Tensor,
        cu_seqlens_q_decode: zml.Tensor,
        seqused_k_decode: zml.Tensor,

        metadata: MixedMetadata,
        options: MixedOptions,

        pub fn init(options: MixedOptions) MixedParameters {
            return .{
                .block_table_prefill = .init(.{ .b = options.batch_size_prefill, .p = options.max_num_pages }, .i32),
                .cu_seqlens_q_prefill = .init(.{ .b = options.batch_size_prefill + 1 }, .i32),
                .seqused_k_prefill = .init(.{ .b = options.batch_size_prefill }, .i32),
                .block_table_decode = .init(.{ .b = options.batch_size_decode, .p = options.max_num_pages }, .i32),
                .cu_seqlens_q_decode = .init(.{ .b = options.batch_size_decode + 1 }, .i32),
                .seqused_k_decode = .init(.{ .b = options.batch_size_decode }, .i32),
                .metadata = MixedMetadata.init(options),
                .options = options,
            };
        }

        pub fn allocationSize(self: MixedParameters) usize {
            var allocation_size: usize = 0;
            allocation_size += self.block_table_decode.byteSize();
            allocation_size += self.cu_seqlens_q_prefill.byteSize();
            allocation_size += self.seqused_k_prefill.byteSize();

            allocation_size += self.block_table_decode.byteSize();
            allocation_size += self.cu_seqlens_q_decode.byteSize();
            allocation_size += self.seqused_k_decode.byteSize();
            allocation_size += self.metadata.allocationSize();

            return allocation_size;
        }
    };

    pub const MixedMetadata = struct {
        out_accum: zml.Tensor,
        host_metadata: zml.Tensor,

        pub fn init(opts: MixedOptions) MixedMetadata {
            const shape = zml.Shape.init(.{
                .splits = MAX_NUM_SPLITS,
                .hkv = opts.num_kv_heads,
                .hg = @divExact(opts.num_heads, opts.num_kv_heads),
                .b = opts.max_token_count,
                .hd = opts.head_dim,
            }, .f32).withPartitioning(.{ .hkv = .model });
            return .{
                .out_accum = .fromShape(shape),
                .host_metadata = .init(.{2}, .i32),
            };
        }

        pub fn initBuffers(self: *const MixedMetadata, io: std.Io, platform: zml.Platform, out_accum_sharding: zml.sharding.Sharding) !zml.Bufferized(MixedMetadata) {
            return .{
                .out_accum = try zml.Buffer.uninitialized(io, &platform, self.out_accum.shape(), out_accum_sharding, .{}),
                .host_metadata = undefined,
            };
        }

        pub fn deinitBuffers(self: *const zml.Bufferized(MixedMetadata)) void {
            self.out_accum.deinit();
        }

        pub fn allocationSize(self: MixedMetadata) usize {
            var allocation_size: usize = 0;
            allocation_size += self.out_accum.byteSize();
            allocation_size += self.host_metadata.byteSize();
            return allocation_size;
        }
    };

    pub const Context = struct {
        max_seqlen_k: usize,
        decode_offset: ?zml.Tensor = null,

        pub fn init(parameters: Parameters, num_heads: i64, num_kv_heads: i64, head_dim: i64, page_size: i64) Context {
            _ = num_heads;
            _ = num_kv_heads;
            _ = head_dim;
            _ = page_size;
            const max_seqlen_k = switch (parameters) {
                inline else => |v| v.options.max_seqlen_k,
            };
            const decode_offset = switch (parameters) {
                .mixed => |mixed_parameters| mixed_parameters.metadata.host_metadata.slice1d(0, .{ .end = 1 }).squeeze(0),
                .decode => null,
            };
            return .{ .max_seqlen_k = max_seqlen_k, .decode_offset = decode_offset };
        }
    };

    pub const Prefill = struct {
        pub const custom_call_name = "paged_fa3_prefill";
        const Wrapped = Wrapper(@This(), .runInner);

        const register = Wrapped.register;
        const run = Wrapped.run;

        pub fn runInner(call_frame: *ffi.CallFrame) !?*ffi.Error {
            if (call_frame.registeringHook()) return null;

            const layer_index = bufferFromFfiBuffer(call_frame.args.buffers()[11]);
            const layer_index_raw = @as([*]i32, @ptrCast(@alignCast(layer_index.ptr)))[0];

            const q = bufferFromFfiBuffer(call_frame.args.buffers()[0]);
            const paged_k = fixupKvCacheBuffer(bufferFromFfiBuffer(call_frame.args.buffers()[1]), layer_index_raw);
            const paged_v = fixupKvCacheBuffer(bufferFromFfiBuffer(call_frame.args.buffers()[2]), layer_index_raw);
            const cu_seqlens_q = bufferFromFfiBuffer(call_frame.args.buffers()[3]);
            const seqused_k = bufferFromFfiBuffer(call_frame.args.buffers()[4]);
            const block_table = bufferFromFfiBuffer(call_frame.args.buffers()[5]);
            const softmax_lse = bufferFromFfiBuffer(call_frame.args.buffers()[6]);
            const softmax_lse_accum = bufferFromFfiBuffer(call_frame.args.buffers()[7]);
            const out_accum = bufferFromFfiBuffer(call_frame.args.buffers()[8]);
            const scheduler_metadata = bufferFromFfiBuffer(call_frame.args.buffers()[9]);
            const host_metadata = bufferFromFfiBuffer(call_frame.args.buffers()[10]);
            const o = bufferFromFfiBuffer(call_frame.results.buffers()[0]);

            const max_seqlen_q = @as([*]i32, @ptrCast(@alignCast(host_metadata.ptr)))[1];
            const softmax_scale: f32 = getScalarAttributeAs(f32, call_frame, "softmax_scale") orelse b: {
                const head_dim = q.shape.dim(2);
                break :b 1.0 / std.math.sqrt(@as(f32, @floatFromInt(head_dim)));
            };
            const is_causal: bool = getScalarAttributeAs(bool, call_frame, "is_causal").?;
            const window_size_left: i32 = getScalarAttributeAs(i32, call_frame, "window_size_left") orelse -1;
            const window_size_right: i32 = getScalarAttributeAs(i32, call_frame, "window_size_right") orelse -1;
            const max_seqlen_k: i32 = getScalarAttributeAs(i32, call_frame, "max_seqlen_k").?;

            const ctx: *ffi.ExecutionContext = @constCast(call_frame.ctx);
            const stream = call_frame.api.stream(ctx);

            const params: flashattn.FA3MhaFwdParams = .{
                .max_seqlen_q = max_seqlen_q,
                .max_seqlen_k = max_seqlen_k,
                .is_causal = is_causal,
                .softmax_scale = softmax_scale,
                .window_size_left = window_size_left,
                .window_size_right = window_size_right,
                .softcap = 0.0,
                .is_rotary_interleaved = false,
                .num_splits = MAX_NUM_SPLITS,
                .sm_margin = 0,
                .cp_world_size = 1,
                .cp_rank = 0,
            };

            flashattn.fa3_mha_fwd(
                &q.toFlashattnTensor(),
                &paged_k.toFlashattnTensor(),
                &paged_v.toFlashattnTensor(),
                &o.toFlashattnTensor(),
                &cu_seqlens_q.toFlashattnTensor(),
                null,
                null,
                &seqused_k.toFlashattnTensor(),
                &block_table.toFlashattnTensor(),
                null,
                null,
                null,
                &softmax_lse.toFlashattnTensor(),
                &softmax_lse_accum.toFlashattnTensor(),
                &out_accum.toFlashattnTensor(),
                &scheduler_metadata.toFlashattnTensor(),
                null,
                null,
                &params,
                stream,
            );

            return null;
        }
    };

    pub const Decode = struct {
        pub const custom_call_name = "paged_fa3_decode";
        const Wrapped = Wrapper(@This(), .runInner);

        const register = Wrapped.register;
        const run = Wrapped.run;

        pub fn runInner(call_frame: *ffi.CallFrame) !?*ffi.Error {
            if (call_frame.registeringHook()) return null;

            const layer_index = bufferFromFfiBuffer(call_frame.args.buffers()[10]);
            const layer_index_raw = @as([*]i32, @ptrCast(@alignCast(layer_index.ptr)))[0];

            const q = bufferFromFfiBuffer(call_frame.args.buffers()[0]);
            const paged_k = fixupKvCacheBuffer(bufferFromFfiBuffer(call_frame.args.buffers()[1]), layer_index_raw);
            const paged_v = fixupKvCacheBuffer(bufferFromFfiBuffer(call_frame.args.buffers()[2]), layer_index_raw);
            const cu_seqlens_q = bufferFromFfiBuffer(call_frame.args.buffers()[3]);
            const seqused_k = bufferFromFfiBuffer(call_frame.args.buffers()[4]);
            const block_table = bufferFromFfiBuffer(call_frame.args.buffers()[5]);
            const softmax_lse = bufferFromFfiBuffer(call_frame.args.buffers()[6]);
            const softmax_lse_accum = bufferFromFfiBuffer(call_frame.args.buffers()[7]);
            const out_accum = bufferFromFfiBuffer(call_frame.args.buffers()[8]);
            const scheduler_metadata = bufferFromFfiBuffer(call_frame.args.buffers()[9]);
            const o = bufferFromFfiBuffer(call_frame.results.buffers()[0]);

            const softmax_scale: f32 = getScalarAttributeAs(f32, call_frame, "softmax_scale") orelse b: {
                const head_dim = q.shape.dim(2);
                break :b 1.0 / std.math.sqrt(@as(f32, @floatFromInt(head_dim)));
            };
            const is_causal: bool = getScalarAttributeAs(bool, call_frame, "is_causal").?;
            const window_size_left: i32 = getScalarAttributeAs(i32, call_frame, "window_size_left") orelse -1;
            const window_size_right: i32 = getScalarAttributeAs(i32, call_frame, "window_size_right") orelse -1;
            const max_seqlen_k: i32 = getScalarAttributeAs(i32, call_frame, "max_seqlen_k").?;

            const ctx: *ffi.ExecutionContext = @constCast(call_frame.ctx);
            const stream = call_frame.api.stream(ctx);

            const params: flashattn.FA3MhaFwdParams = .{
                .max_seqlen_q = 1,
                .max_seqlen_k = max_seqlen_k,
                .is_causal = is_causal,
                .softmax_scale = softmax_scale,
                .window_size_left = window_size_left,
                .window_size_right = window_size_right,
                .softcap = 0.0,
                .is_rotary_interleaved = false,
                .num_splits = MAX_NUM_SPLITS,
                .sm_margin = 0,
                .cp_world_size = 1,
                .cp_rank = 0,
            };

            flashattn.fa3_mha_fwd(
                &q.toFlashattnTensor(),
                &paged_k.toFlashattnTensor(),
                &paged_v.toFlashattnTensor(),
                &o.toFlashattnTensor(),
                &cu_seqlens_q.toFlashattnTensor(),
                null,
                null,
                &seqused_k.toFlashattnTensor(),
                &block_table.toFlashattnTensor(),
                null,
                null,
                null,
                &softmax_lse.toFlashattnTensor(),
                &softmax_lse_accum.toFlashattnTensor(),
                &out_accum.toFlashattnTensor(),
                &scheduler_metadata.toFlashattnTensor(),
                null,
                null,
                &params,
                stream,
            );

            return null;
        }
    };

    pub fn pagedAttention(parameters: Parameters, context: Context, q: zml.Tensor, k_cache: zml.Tensor, v_cache: zml.Tensor, layer_index: zml.Tensor, opts: AttentionOptions) zml.Tensor {
        stdx.debug.assert(q.shape().hasTags(.{ .b, .hg, .hkv, .hd }), "Expected q to have tags .b, .h, .hd", .{});
        stdx.debug.assert(k_cache.shape().hasTags(.{ .page, .k_chunk, .hkv, .hd }), "Expected paged_k to have tags .page, .k_chunk, .h, .hd, got {}", .{k_cache.shape()});
        stdx.debug.assert(v_cache.shape().hasTags(.{ .page, .k_chunk, .hkv, .hd }), "Expected paged_v to have tags .page, .k_chunk, .h, .hd. got {}", .{v_cache.shape()});
        if (q.dim(.hd) > 256) {
            return pagedAttentionVanilla(parameters, context, q, k_cache, v_cache, layer_index, opts);
        }

        const num_head_groups = q.dim(.hg);
        const num_kv_heads = q.dim(.hkv);
        const o = switch (parameters) {
            .decode => |decode_parameters| b: {
                const batch_size = decode_parameters.block_table.dim(0);

                const block_table = decode_parameters.block_table.withPartitioning(.{ .b = .replicated });
                const cu_seqlens_q = decode_parameters.cu_seqlens_q.withPartitioning(.{ .b = .replicated });
                const seqused_k = decode_parameters.seqused_k.withPartitioning(.{ .b = .replicated });
                const out_accum = decode_parameters.metadata.out_accum.withPartitioning(.{ .hkv = .model });

                const softmax_lse = zml.Tensor.constant(zml.DataType.f32.zero()).broad(.init(.{
                    .hkv = num_kv_heads,
                    .hg = num_head_groups,
                    .q = q.dim(.b),
                }, .f32)).withPartitioning(.{ .hkv = .model });
                const softmax_lse_accum = zml.Tensor.constant(zml.DataType.f32.zero()).broad(.init(.{
                    .splits = MAX_NUM_SPLITS,
                    .q = q.dim(.b),
                    .hkv = num_kv_heads,
                    .hg = num_head_groups,
                }, .f32)).withPartitioning(.{ .hkv = .model });
                const scheduler_metadata = zml.Tensor.constant(zml.DataType.i32.zero()).broad(.init(.{ .b = batch_size + 1 }, .i32)).withPartitioning(.{ .b = .replicated });

                var q2 = q.merge(.{ .h = .{ .hkv, .hg } }).withPartitioning(.{ .h = .model });

                const output_shape = q2.shape();
                var o = zml.ops.manualComputation(
                    .{
                        q2,
                        k_cache,
                        v_cache,
                        cu_seqlens_q,
                        seqused_k,
                        block_table,
                        softmax_lse,
                        softmax_lse_accum,
                        out_accum,
                        scheduler_metadata,
                        layer_index,
                    },
                    output_shape,
                    .{
                        .metadata = .{
                            .is_causal = opts.is_causal,
                            .max_seqlen_k = context.max_seqlen_k,
                            .window_size_left = opts.sliding_window,
                        },
                        .opts = zml.ops.CustomCallOptions{
                            .has_side_effect = false,
                        },
                    },
                    (struct {
                        fn body(ctx_: anytype, _: std.mem.Allocator, sharded_inputs: []const zml.Tensor, output: zml.Shape) zml.Tensor {
                            return zml.ops.customCall(Decode.custom_call_name, sharded_inputs, output, ctx_.metadata, ctx_.opts);
                        }
                    }).body,
                );

                o = o.splitAxis(.h, .{ .hkv = num_kv_heads, .hg = num_head_groups });

                break :b o;
            },
            .mixed => |mixed_parameters| b: {
                const batch_size_prefill = mixed_parameters.block_table_prefill.dim(0);

                const block_table_prefill = mixed_parameters.block_table_prefill.withPartitioning(.{ .b = .replicated });
                const cu_seqlens_q_prefill = mixed_parameters.cu_seqlens_q_prefill.withPartitioning(.{ .b = .replicated });
                const seqused_k_prefill = mixed_parameters.seqused_k_prefill.withPartitioning(.{ .b = .replicated });
                const block_table_decode = mixed_parameters.block_table_decode.withPartitioning(.{ .b = .replicated });
                const cu_seqlens_q_decode = mixed_parameters.cu_seqlens_q_decode.withPartitioning(.{ .b = .replicated });
                const seqused_k_decode = mixed_parameters.seqused_k_decode.withPartitioning(.{ .b = .replicated });
                const out_accum = mixed_parameters.metadata.out_accum.withPartitioning(.{ .hkv = .model });
                const host_metadata = mixed_parameters.metadata.host_metadata.withTags(.{.x}).withPartitioning(.{ .x = .replicated });

                const softmax_lse_prefill = zml.Tensor.constant(zml.DataType.f32.zero()).broad(.init(.{
                    .hkv = num_kv_heads,
                    .hg = num_head_groups,
                    .q = q.dim(.b),
                }, .f32)).withPartitioning(.{ .hkv = .model });
                const softmax_lse_accum_prefill = zml.Tensor.constant(zml.DataType.f32.zero()).broad(.init(.{
                    .splits = MAX_NUM_SPLITS,
                    .q = q.dim(.b),
                    .hkv = num_kv_heads,
                    .hg = num_head_groups,
                }, .f32)).withPartitioning(.{ .hkv = .model });
                const scheduler_metadata_prefill = zml.Tensor.constant(zml.DataType.i32.zero()).broad(.init(.{ .b = batch_size_prefill + 1 }, .i32)).withPartitioning(.{ .b = .replicated });

                var q2 = q.merge(.{ .h = .{ .hkv, .hg } }).withPartitioning(.{ .h = .model });

                const output_shape = q2.shape();
                var o = zml.ops.manualComputation(
                    .{
                        q2,
                        k_cache,
                        v_cache,
                        cu_seqlens_q_prefill,
                        seqused_k_prefill,
                        block_table_prefill,
                        softmax_lse_prefill,
                        softmax_lse_accum_prefill,
                        out_accum,
                        scheduler_metadata_prefill,
                        host_metadata,
                        layer_index,
                    },
                    output_shape,
                    .{
                        .metadata = .{
                            .is_causal = opts.is_causal,
                            .max_seqlen_k = context.max_seqlen_k,
                            .window_size_left = opts.sliding_window,
                        },
                        .opts = zml.ops.CustomCallOptions{ .has_side_effect = false },
                    },
                    (struct {
                        fn body(ctx_: anytype, _: std.mem.Allocator, sharded_inputs: []const zml.Tensor, output: zml.Shape) zml.Tensor {
                            return zml.ops.customCall(Prefill.custom_call_name, sharded_inputs, output, ctx_.metadata, ctx_.opts);
                        }
                    }).body,
                );

                o = o.splitAxis(.h, .{ .hkv = num_kv_heads, .hg = num_head_groups });

                const batch_size_decode = mixed_parameters.block_table_prefill.dim(0);
                const softmax_lse_decode = zml.Tensor.constant(zml.DataType.f32.zero()).broad(.init(.{
                    .hkv = num_kv_heads,
                    .hg = num_head_groups,
                    .q = q.dim(.b),
                }, .f32));
                const softmax_lse_accum_decode = zml.Tensor.constant(zml.DataType.f32.zero()).broad(.init(.{
                    .splits = MAX_NUM_SPLITS,
                    .q = q.dim(.b),
                    .hkv = num_kv_heads,
                    .hg = num_head_groups,
                }, .f32));
                const scheduler_metadata_decode = zml.Tensor.constant(zml.DataType.i32.zero()).broad(.init(.{ .b = batch_size_decode + 1 }, .i32)).withPartitioning(.{ .b = .replicated });
                var q_decode = q.dynamicSlice1d(0, .{ .start = context.decode_offset.?, .len = batch_size_decode }).withPartitioning(.{ .hkv = .model });

                q_decode = q_decode.merge(.{ .h = .{ .hkv, .hg } }).withPartitioning(.{ .h = .model });

                const decode_output_shape = q_decode.shape();
                var o_decode = zml.ops.manualComputation(
                    .{
                        q_decode,
                        k_cache,
                        v_cache,
                        cu_seqlens_q_decode,
                        seqused_k_decode,
                        block_table_decode,
                        softmax_lse_decode,
                        softmax_lse_accum_decode,
                        out_accum,
                        scheduler_metadata_decode,
                        layer_index,
                    },
                    decode_output_shape,
                    .{
                        .metadata = .{
                            .is_causal = opts.is_causal,
                            .max_seqlen_k = context.max_seqlen_k,
                            .window_size_left = opts.sliding_window,
                        },
                        .opts = zml.ops.CustomCallOptions{ .has_side_effect = false },
                    },
                    (struct {
                        fn body(ctx_: anytype, _: std.mem.Allocator, sharded_inputs: []const zml.Tensor, output: zml.Shape) zml.Tensor {
                            return zml.ops.customCall(Decode.custom_call_name, sharded_inputs, output, ctx_.metadata, ctx_.opts);
                        }
                    }).body,
                );
                o_decode = o_decode.splitAxis(.h, .{ .hkv = num_kv_heads, .hg = num_head_groups });

                o = o.dynamicUpdateSlice1d(o_decode, 0, context.decode_offset.?);
                break :b o;
            },
        };

        return o;
    }
};
