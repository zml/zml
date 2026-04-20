const std = @import("std");

const flashattn = @import("platforms/cuda/flashattn");
const platforms = @import("platforms");
const stdx = @import("stdx");

const CompilationContext = @import("../module.zig").CompilationContext;
const zml = @import("../zml.zig");
const AttentionOptions = @import("paged_attention.zig").AttentionOptions;

pub fn load(allocator: std.mem.Allocator, io: std.Io) !void {
    if (comptime platforms.isEnabled(.cuda)) {
        try flashattn.load(allocator, io);
    }
}

pub fn register(platform: *const zml.Platform) !void {
    if (comptime platforms.isEnabled(.cuda)) {
        try fa3.fa3_mha_fwd.register(platform);
        try fa2.fa2_mha_varlen_fwd.register(platform);
        try paged_fa2.Decode.paged_fa2_decode.register(platform);
        try paged_fa2.Prefill.paged_fa2_prefill.register(platform);
        try paged_fa3.Decode.paged_fa3_decode.register(platform);
        try paged_fa3.Prefill.paged_fa3_prefill.register(platform);
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

fn toFlashattnTensor(buffer: zml.pjrtx.CustomCallBuffer) flashattn.Tensor {
    return .init(
        buffer.ptr,
        buffer.shape.dims(),
        buffer.shape.withDtype(.u8).computeByteStrides().constSlice(),
        flashattnDataTypeFromZmlDataType(buffer.shape.dtype()),
    );
}

pub const fa2 = struct {
    const Input = struct {
        q: zml.Tensor,
        k: zml.Tensor,
        v: zml.Tensor,
        cu_seqlens_q: zml.Tensor,
        cu_seqlens_k: zml.Tensor,
        softmax_lse: zml.Tensor,
        softmax_lse_accum: zml.Tensor,
        out_accum: zml.Tensor,
    };

    const Output = struct {
        o: zml.Shape,
    };

    const Attributes = struct {
        softmax_scale: f32,
        is_causal: bool,
        window_size_left: i32,
        window_size_right: i32,
        max_seqlen_q: i32,
        max_seqlen_k: i32,
        num_heads: i32,
    };

    fn ffiCall(
        call_frame: *zml.pjrt.ffi.CallFrame,
        input: zml.ops.TensorToCustomCallBuffer(Input),
        output: zml.ops.ShapeToCustomCallBuffer(Output),
        attributes: Attributes,
    ) !?*zml.pjrt.ffi.Error {
        const params: flashattn.FA2MhaVarlenFwdParams = .{
            .max_seqlen_q = attributes.max_seqlen_q,
            .max_seqlen_k = attributes.max_seqlen_k,
            .is_causal = attributes.is_causal,
            .softmax_scale = attributes.softmax_scale,
            .window_size_left = attributes.window_size_left,
            .window_size_right = attributes.window_size_right,
            .num_splits = 0,
            .num_heads = attributes.num_heads,
        };

        const stream = call_frame.api.stream(call_frame.ctx);

        flashattn.fa2_mha_varlen_fwd(
            &toFlashattnTensor(input.q),
            &toFlashattnTensor(input.k),
            &toFlashattnTensor(input.v),
            &toFlashattnTensor(output.o),
            &toFlashattnTensor(input.cu_seqlens_q),
            &toFlashattnTensor(input.cu_seqlens_k),
            null,
            null,
            &toFlashattnTensor(input.softmax_lse),
            null,
            &toFlashattnTensor(input.softmax_lse_accum),
            &toFlashattnTensor(input.out_accum),
            &params,
            stream,
        );

        return null;
    }

    const fa2_mha_varlen_fwd = zml.ops.CustomCall(Input, Output, Attributes, ffiCall, .{
        .name = "fa2_mha_varlen_fwd",
        .sharding_aware = true,
        .has_side_effect = false,
        .output_operand_aliases = .{ .o = .q },
    });

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

        const output = fa2_mha_varlen_fwd.call(
            .{
                .q = q_sharded,
                .k = k,
                .v = v,
                .cu_seqlens_q = cu_seqlens_q,
                .cu_seqlens_k = cu_seqlens_k.withTags(.{.i}).withPartitioning(.{ .i = .replicated }),
                .softmax_lse = metadata.softmax_lse.withPartitioning(.{ .h = .model }),
                .softmax_lse_accum = metadata.softmax_lse_accum.withPartitioning(.{ .h = .model }),
                .out_accum = metadata.out_accum.withPartitioning(.{ .h = .model }),
            },
            .{
                .o = q_sharded.shape(),
            },
            .{
                .softmax_scale = b: {
                    const head_dim = q.shape().dim(2);
                    break :b 1.0 / std.math.sqrt(@as(f32, @floatFromInt(head_dim)));
                },
                .is_causal = true,
                .window_size_left = @as(i32, -1),
                .window_size_right = @as(i32, -1),
                .max_seqlen_q = max_seqlen_q,
                .max_seqlen_k = max_seqlen_k,
                .num_heads = @as(i32, @intCast(@divExact(num_heads, model_partitions))),
            },
        );
        var o = output.o;

        if (seqlenq_ngroups_swapped) {
            o = o.splitAxis(.tot, .{ .tot = original_tot, .ngroups = ngroups }).transpose(.{ .tot, .h, .ngroups, .hd }).merge(.{ .h = .{ .h, .ngroups } });
        }

        return o.splitAxis(.tot, .{ .b = 1, .q = q_.dim(.q) }).squeeze(.b);
    }
};

pub const fa3 = struct {
    const Input = struct {
        q: zml.Tensor,
        k: zml.Tensor,
        v: zml.Tensor,
        cu_seqlens_q: zml.Tensor,
        cu_seqlens_k: zml.Tensor,
        softmax_lse: zml.Tensor,
        softmax_lse_accum: zml.Tensor,
        out_accum: zml.Tensor,
        scheduler_metadata: zml.Tensor,
    };

    const Output = struct {
        o: zml.Shape,
    };

    const Attributes = struct {
        softmax_scale: f32,
        is_causal: bool,
        window_size_left: i32,
        window_size_right: i32,
        max_seqlen_q: i32,
        max_seqlen_k: i32,
    };

    fn ffiCall(
        call_frame: *zml.pjrt.ffi.CallFrame,
        input: zml.ops.TensorToCustomCallBuffer(Input),
        output: zml.ops.ShapeToCustomCallBuffer(Output),
        attributes: Attributes,
    ) !?*zml.pjrt.ffi.Error {
        const params: flashattn.FA3MhaFwdParams = .{
            .max_seqlen_q = attributes.max_seqlen_q,
            .max_seqlen_k = attributes.max_seqlen_k,
            .softcap = 0.0,
            .is_rotary_interleaved = false,
            .num_splits = 0,
            .sm_margin = 0,
            .is_causal = attributes.is_causal,
            .softmax_scale = attributes.softmax_scale,
            .window_size_left = attributes.window_size_left,
            .window_size_right = attributes.window_size_right,
            .cp_world_size = 1,
            .cp_rank = 0,
        };

        const stream = call_frame.api.stream(call_frame.ctx);

        flashattn.fa3_mha_fwd(
            &toFlashattnTensor(input.q),
            &toFlashattnTensor(input.k),
            &toFlashattnTensor(input.v),
            &toFlashattnTensor(output.o),
            &toFlashattnTensor(input.cu_seqlens_q),
            &toFlashattnTensor(input.cu_seqlens_k),
            null,
            null,
            null,
            null,
            null,
            null,
            &toFlashattnTensor(input.softmax_lse),
            &toFlashattnTensor(input.softmax_lse_accum),
            &toFlashattnTensor(input.out_accum),
            &toFlashattnTensor(input.scheduler_metadata),
            null,
            null,
            &params,
            stream,
        );

        return null;
    }

    const fa3_mha_fwd = zml.ops.CustomCall(Input, Output, Attributes, ffiCall, .{
        .name = "fa3_mha_fwd",
        .sharding_aware = true,
        .has_side_effect = false,
        .output_operand_aliases = .{ .o = .q },
    });

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
        const q = q_.insertAxes(.q, .{.b}).merge(.{ .tot = .{ .b, .q } });
        const k = k_.insertAxes(.k, .{.b}).merge(.{ .tot = .{ .b, .k } });
        const v = v_.insertAxes(.k, .{.b}).merge(.{ .tot = .{ .b, .k } });
        // TODO(Corendos): replace with cumsum
        const cu_seqlens_q = zml.Tensor.constantTensor(zml.Shape.init(.{2}, .i32), std.mem.sliceAsBytes(&[2]i32{ 0, max_seqlen_q }))
            .withPartitioning(.{ ._0 = .replicated });

        const output = fa3_mha_fwd.call(
            .{
                .q = q,
                .k = k,
                .v = v,
                .cu_seqlens_q = cu_seqlens_q,
                .cu_seqlens_k = cu_seqlens_k.withTags(.{.i}).withPartitioning(.{ .i = .replicated }),
                .softmax_lse = metadata.softmax_lse.withPartitioning(.{ .h = .model }),
                .softmax_lse_accum = metadata.softmax_lse_accum.withPartitioning(.{ .h = .model }),
                .out_accum = metadata.out_accum.withPartitioning(.{ .h = .model }),
                .scheduler_metadata = metadata.scheduler_metadata.withPartitioning(.{ .meta = .replicated }),
            },
            .{
                .o = q.shape(),
            },
            .{
                .softmax_scale = b: {
                    const head_dim = q.shape().dim(2);
                    break :b 1.0 / std.math.sqrt(@as(f32, @floatFromInt(head_dim)));
                },
                .is_causal = true,
                .window_size_left = @as(i32, -1),
                .window_size_right = @as(i32, -1),
                .max_seqlen_q = max_seqlen_q,
                .max_seqlen_k = max_seqlen_k,
            },
        );

        return output.o.splitAxis(.tot, .{ .b = 1, .q = q_.dim(.q) }).squeeze(.b);
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
        const Input = struct {
            q: zml.Tensor,
            paged_k: zml.Tensor,
            paged_v: zml.Tensor,
            cu_seqlens_q: zml.Tensor,
            cu_seqlens_k: zml.Tensor,
            seqused_k: zml.Tensor,
            block_table: zml.Tensor,
            softmax_lse: zml.Tensor,
            softmax_lse_accum: zml.Tensor,
            out_accum: zml.Tensor,
            host_metadata: zml.Tensor,
        };

        const Output = struct {
            o: zml.Shape,
        };

        const PrefillAttributes = struct {
            softmax_scale: f32,
            is_causal: bool,
            window_size_left: i32,
            window_size_right: i32,
            max_seqlen_k: i32,
            num_heads: i32,
        };

        fn ffiCall(
            call_frame: *zml.pjrt.ffi.CallFrame,
            input: zml.ops.TensorToCustomCallBuffer(Input),
            output: zml.ops.ShapeToCustomCallBuffer(Output),
            attributes: PrefillAttributes,
        ) !?*zml.pjrt.ffi.Error {
            const max_seqlen_q = @as([*]i32, @ptrCast(@alignCast(input.host_metadata.ptr)))[1];

            const params: flashattn.FA2MhaVarlenFwdParams = .{
                .max_seqlen_q = max_seqlen_q,
                .max_seqlen_k = attributes.max_seqlen_k,
                .is_causal = attributes.is_causal,
                .softmax_scale = attributes.softmax_scale,
                .window_size_left = attributes.window_size_left,
                .window_size_right = attributes.window_size_right,
                .num_splits = MAX_NUM_SPLITS,
                .num_heads = attributes.num_heads,
            };

            const stream = call_frame.api.stream(call_frame.ctx);

            flashattn.fa2_mha_varlen_fwd(
                &toFlashattnTensor(input.q),
                &toFlashattnTensor(input.paged_k),
                &toFlashattnTensor(input.paged_v),
                &toFlashattnTensor(output.o),
                &toFlashattnTensor(input.cu_seqlens_q),
                &toFlashattnTensor(input.cu_seqlens_k),
                &toFlashattnTensor(input.seqused_k),
                &toFlashattnTensor(input.block_table),
                &toFlashattnTensor(input.softmax_lse),
                null,
                &toFlashattnTensor(input.softmax_lse_accum),
                &toFlashattnTensor(input.out_accum),
                &params,
                stream,
            );

            return null;
        }

        const paged_fa2_prefill = zml.ops.CustomCall(Input, Output, PrefillAttributes, ffiCall, .{
            .name = "paged_fa2_prefill",
            .sharding_aware = true,
            .has_side_effect = false,
            .output_operand_aliases = .{ .o = .q },
        });

        pub const register = paged_fa2_prefill.register;
    };

    pub const Decode = struct {
        const Input = struct {
            q: zml.Tensor,
            paged_k: zml.Tensor,
            paged_v: zml.Tensor,
            cu_seqlens_q: zml.Tensor,
            cu_seqlens_k: zml.Tensor,
            seqused_k: zml.Tensor,
            block_table: zml.Tensor,
            softmax_lse: zml.Tensor,
            softmax_lse_accum: zml.Tensor,
            out_accum: zml.Tensor,
        };

        const Output = struct {
            o: zml.Shape,
        };

        const DecodeAttributes = struct {
            softmax_scale: f32,
            is_causal: bool,
            window_size_left: i32,
            window_size_right: i32,
            max_seqlen_k: i32,
            num_heads: i32,
        };

        fn ffiCall(
            call_frame: *zml.pjrt.ffi.CallFrame,
            input: zml.ops.TensorToCustomCallBuffer(Input),
            output: zml.ops.ShapeToCustomCallBuffer(Output),
            attributes: DecodeAttributes,
        ) !?*zml.pjrt.ffi.Error {
            const params: flashattn.FA2MhaVarlenFwdParams = .{
                .max_seqlen_q = 1,
                .max_seqlen_k = attributes.max_seqlen_k,
                .is_causal = attributes.is_causal,
                .softmax_scale = attributes.softmax_scale,
                .window_size_left = attributes.window_size_left,
                .window_size_right = attributes.window_size_right,
                .num_splits = MAX_NUM_SPLITS,
                .num_heads = attributes.num_heads,
            };

            const stream = call_frame.api.stream(call_frame.ctx);

            flashattn.fa2_mha_varlen_fwd(
                &toFlashattnTensor(input.q),
                &toFlashattnTensor(input.paged_k),
                &toFlashattnTensor(input.paged_v),
                &toFlashattnTensor(output.o),
                &toFlashattnTensor(input.cu_seqlens_q),
                &toFlashattnTensor(input.cu_seqlens_k),
                &toFlashattnTensor(input.seqused_k),
                &toFlashattnTensor(input.block_table),
                &toFlashattnTensor(input.softmax_lse),
                null,
                &toFlashattnTensor(input.softmax_lse_accum),
                &toFlashattnTensor(input.out_accum),
                &params,
                stream,
            );

            return null;
        }

        const paged_fa2_decode = zml.ops.CustomCall(Input, Output, DecodeAttributes, ffiCall, .{
            .name = "paged_fa2_decode",
            .sharding_aware = true,
            .has_side_effect = false,
            .output_operand_aliases = .{ .o = .q },
        });

        pub const register = paged_fa2_decode.register;
    };

    pub fn pagedAttention(parameters: Parameters, context: Context, q: zml.Tensor, k_cache: zml.Tensor, v_cache: zml.Tensor, opts: AttentionOptions) zml.Tensor {
        stdx.debug.assert(q.shape().hasTags(.{ .b, .hg, .hkv, .hd }), "Expected q to have tags .b, .h, .hd", .{});
        stdx.debug.assert(k_cache.shape().hasTags(.{ .page, .k_chunk, .hkv, .hd }), "Expected paged_k to have tags .page, .k_chunk, .h, .hd, got {}", .{k_cache.shape()});
        stdx.debug.assert(v_cache.shape().hasTags(.{ .page, .k_chunk, .hkv, .hd }), "Expected paged_v to have tags .page, .k_chunk, .h, .hd. got {}", .{v_cache.shape()});
        const ctx = CompilationContext.current();

        const num_head_groups = q.dim(.hg);
        const num_kv_heads = q.dim(.hkv);
        const head_dim = q.dim(.hd);
        const num_heads = num_head_groups * num_kv_heads;
        // FIXME: remove unreachable and propagate error correctly.
        const num_heads_per_shard = @divExact(num_heads, ctx.partitioning.numPartitionsForLogicalAxis(q.shape(), .model) catch unreachable);

        const decode_attributes: Decode.DecodeAttributes = .{
            .is_causal = opts.is_causal,
            .max_seqlen_k = @intCast(context.max_seqlen_k),
            .num_heads = @intCast(num_heads_per_shard),
            .window_size_left = opts.sliding_window,
            .window_size_right = @as(i32, -1),
            .softmax_scale = opts.scale,
        };

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

                const output = Decode.paged_fa2_decode.call(
                    .{
                        .q = q2,
                        .paged_k = k_cache,
                        .paged_v = v_cache,
                        .cu_seqlens_q = cu_seqlens_q,
                        .cu_seqlens_k = dummy_cu_seqlens_k,
                        .seqused_k = seqused_k,
                        .block_table = block_table,
                        .softmax_lse = softmax_lse,
                        .softmax_lse_accum = softmax_lse_accum,
                        .out_accum = out_accum,
                    },
                    .{
                        .o = q2.shape(),
                    },
                    decode_attributes,
                );
                var o = output.o;

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

                const prefill_attributes: Prefill.PrefillAttributes = .{
                    .is_causal = opts.is_causal,
                    .max_seqlen_k = @intCast(context.max_seqlen_k),
                    .num_heads = @intCast(num_heads_per_shard),
                    .window_size_left = opts.sliding_window,
                    .window_size_right = @as(i32, -1),
                    .softmax_scale = opts.scale,
                };

                const prefill_output = Prefill.paged_fa2_prefill.call(
                    .{
                        .q = q2,
                        .paged_k = k_cache,
                        .paged_v = v_cache,
                        .cu_seqlens_q = cu_seqlens_q_prefill,
                        .cu_seqlens_k = dummy_cu_seqlens_k_prefill,
                        .seqused_k = seqused_k_prefill,
                        .block_table = block_table_prefill,
                        .softmax_lse = softmax_lse_prefill,
                        .softmax_lse_accum = softmax_lse_accum_prefill,
                        .out_accum = out_accum_prefill,
                        .host_metadata = host_metadata,
                    },
                    .{
                        .o = q2.shape(),
                    },
                    prefill_attributes,
                );

                var o = prefill_output.o.splitAxis(.h, .{ .hkv = num_kv_heads, .hg = num_head_groups });

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

                const decode_output = Decode.paged_fa2_decode.call(
                    .{
                        .q = q_decode,
                        .paged_k = k_cache,
                        .paged_v = v_cache,
                        .cu_seqlens_q = cu_seqlens_q_decode,
                        .cu_seqlens_k = dummy_cu_seqlens_k_decode,
                        .seqused_k = seqused_k_decode,
                        .block_table = block_table_decode,
                        .softmax_lse = softmax_lse_decode,
                        .softmax_lse_accum = softmax_lse_accum_decode,
                        .out_accum = out_accum_decode,
                    },
                    .{
                        .o = q_decode.shape(),
                    },
                    decode_attributes,
                );
                var o_decode = decode_output.o;

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
        const Input = struct {
            q: zml.Tensor,
            paged_k: zml.Tensor,
            paged_v: zml.Tensor,
            cu_seqlens_q: zml.Tensor,
            seqused_k: zml.Tensor,
            block_table: zml.Tensor,
            softmax_lse: zml.Tensor,
            softmax_lse_accum: zml.Tensor,
            out_accum: zml.Tensor,
            scheduler_metadata: zml.Tensor,
            host_metadata: zml.Tensor,
        };

        const Output = struct {
            o: zml.Shape,
        };

        const PrefillAttributes = struct {
            softmax_scale: f32,
            is_causal: bool,
            window_size_left: i32,
            window_size_right: i32,
            max_seqlen_k: i32,
        };

        fn ffiCall(
            call_frame: *zml.pjrt.ffi.CallFrame,
            input: zml.ops.TensorToCustomCallBuffer(Input),
            output: zml.ops.ShapeToCustomCallBuffer(Output),
            attributes: PrefillAttributes,
        ) !?*zml.pjrt.ffi.Error {
            const max_seqlen_q = @as([*]i32, @ptrCast(@alignCast(input.host_metadata.ptr)))[1];

            const params: flashattn.FA3MhaFwdParams = .{
                .max_seqlen_q = max_seqlen_q,
                .max_seqlen_k = attributes.max_seqlen_k,
                .is_causal = attributes.is_causal,
                .softmax_scale = attributes.softmax_scale,
                .window_size_left = attributes.window_size_left,
                .window_size_right = attributes.window_size_right,
                .softcap = 0.0,
                .is_rotary_interleaved = false,
                .num_splits = MAX_NUM_SPLITS,
                .sm_margin = 0,
                .cp_world_size = 1,
                .cp_rank = 0,
            };

            const stream = call_frame.api.stream(call_frame.ctx);

            flashattn.fa3_mha_fwd(
                &toFlashattnTensor(input.q),
                &toFlashattnTensor(input.paged_k),
                &toFlashattnTensor(input.paged_v),
                &toFlashattnTensor(output.o),
                &toFlashattnTensor(input.cu_seqlens_q),
                null,
                null,
                &toFlashattnTensor(input.seqused_k),
                &toFlashattnTensor(input.block_table),
                null,
                null,
                null,
                &toFlashattnTensor(input.softmax_lse),
                &toFlashattnTensor(input.softmax_lse_accum),
                &toFlashattnTensor(input.out_accum),
                &toFlashattnTensor(input.scheduler_metadata),
                null,
                null,
                &params,
                stream,
            );

            return null;
        }

        const paged_fa3_prefill = zml.ops.CustomCall(Input, Output, PrefillAttributes, ffiCall, .{
            .name = "paged_fa3_prefill",
            .sharding_aware = true,
            .has_side_effect = false,
            .output_operand_aliases = .{ .o = .q },
        });

        pub const register = paged_fa3_prefill.register;
    };

    pub const Decode = struct {
        const Input = struct {
            q: zml.Tensor,
            paged_k: zml.Tensor,
            paged_v: zml.Tensor,
            cu_seqlens_q: zml.Tensor,
            seqused_k: zml.Tensor,
            block_table: zml.Tensor,
            softmax_lse: zml.Tensor,
            softmax_lse_accum: zml.Tensor,
            out_accum: zml.Tensor,
            scheduler_metadata: zml.Tensor,
        };

        const Output = struct {
            o: zml.Shape,
        };

        const DecodeAttributes = struct {
            softmax_scale: f32,
            is_causal: bool,
            window_size_left: i32,
            window_size_right: i32,
            max_seqlen_k: i32,
        };

        fn ffiCall(
            call_frame: *zml.pjrt.ffi.CallFrame,
            input: zml.ops.TensorToCustomCallBuffer(Input),
            output: zml.ops.ShapeToCustomCallBuffer(Output),
            attributes: DecodeAttributes,
        ) !?*zml.pjrt.ffi.Error {
            const params: flashattn.FA3MhaFwdParams = .{
                .max_seqlen_q = 1,
                .max_seqlen_k = attributes.max_seqlen_k,
                .is_causal = attributes.is_causal,
                .softmax_scale = attributes.softmax_scale,
                .window_size_left = attributes.window_size_left,
                .window_size_right = attributes.window_size_right,
                .softcap = 0.0,
                .is_rotary_interleaved = false,
                .num_splits = MAX_NUM_SPLITS,
                .sm_margin = 0,
                .cp_world_size = 1,
                .cp_rank = 0,
            };

            const stream = call_frame.api.stream(call_frame.ctx);

            flashattn.fa3_mha_fwd(
                &toFlashattnTensor(input.q),
                &toFlashattnTensor(input.paged_k),
                &toFlashattnTensor(input.paged_v),
                &toFlashattnTensor(output.o),
                &toFlashattnTensor(input.cu_seqlens_q),
                null,
                null,
                &toFlashattnTensor(input.seqused_k),
                &toFlashattnTensor(input.block_table),
                null,
                null,
                null,
                &toFlashattnTensor(input.softmax_lse),
                &toFlashattnTensor(input.softmax_lse_accum),
                &toFlashattnTensor(input.out_accum),
                &toFlashattnTensor(input.scheduler_metadata),
                null,
                null,
                &params,
                stream,
            );

            return null;
        }

        const paged_fa3_decode = zml.ops.CustomCall(Input, Output, DecodeAttributes, ffiCall, .{
            .name = "paged_fa3_decode",
            .sharding_aware = true,
            .has_side_effect = false,
            .output_operand_aliases = .{ .o = .q },
        });

        pub const register = paged_fa3_decode.register;
    };

    pub fn pagedAttention(parameters: Parameters, context: Context, q: zml.Tensor, k_cache: zml.Tensor, v_cache: zml.Tensor, opts: AttentionOptions) zml.Tensor {
        stdx.debug.assert(q.shape().hasTags(.{ .b, .hg, .hkv, .hd }), "Expected q to have tags .b, .h, .hd", .{});
        stdx.debug.assert(k_cache.shape().hasTags(.{ .page, .k_chunk, .hkv, .hd }), "Expected paged_k to have tags .page, .k_chunk, .h, .hd, got {}", .{k_cache.shape()});
        stdx.debug.assert(v_cache.shape().hasTags(.{ .page, .k_chunk, .hkv, .hd }), "Expected paged_v to have tags .page, .k_chunk, .h, .hd. got {}", .{v_cache.shape()});

        const num_head_groups = q.dim(.hg);
        const num_kv_heads = q.dim(.hkv);

        const decode_attributes: Decode.DecodeAttributes = .{
            .is_causal = opts.is_causal,
            .max_seqlen_k = @intCast(context.max_seqlen_k),
            .window_size_left = opts.sliding_window,
            .window_size_right = @as(i32, -1),
            .softmax_scale = opts.scale,
        };

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

                const q2 = q.merge(.{ .h = .{ .hkv, .hg } }).withPartitioning(.{ .h = .model });

                const output = Decode.paged_fa3_decode.call(
                    .{
                        .q = q2,
                        .paged_k = k_cache,
                        .paged_v = v_cache,
                        .cu_seqlens_q = cu_seqlens_q,
                        .seqused_k = seqused_k,
                        .block_table = block_table,
                        .softmax_lse = softmax_lse,
                        .softmax_lse_accum = softmax_lse_accum,
                        .out_accum = out_accum,
                        .scheduler_metadata = scheduler_metadata,
                    },
                    .{
                        .o = q2.shape(),
                    },
                    decode_attributes,
                );

                break :b output.o.splitAxis(.h, .{ .hkv = num_kv_heads, .hg = num_head_groups });
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

                const q2 = q.merge(.{ .h = .{ .hkv, .hg } }).withPartitioning(.{ .h = .model });

                const prefill_attributes: Prefill.PrefillAttributes = .{
                    .is_causal = opts.is_causal,
                    .max_seqlen_k = @intCast(context.max_seqlen_k),
                    .window_size_left = opts.sliding_window,
                    .window_size_right = @as(i32, -1),
                    .softmax_scale = opts.scale,
                };

                const prefill_output = Prefill.paged_fa3_prefill.call(
                    .{
                        .q = q2,
                        .paged_k = k_cache,
                        .paged_v = v_cache,
                        .cu_seqlens_q = cu_seqlens_q_prefill,
                        .seqused_k = seqused_k_prefill,
                        .block_table = block_table_prefill,
                        .softmax_lse = softmax_lse_prefill,
                        .softmax_lse_accum = softmax_lse_accum_prefill,
                        .out_accum = out_accum,
                        .scheduler_metadata = scheduler_metadata_prefill,
                        .host_metadata = host_metadata,
                    },
                    .{
                        .o = q2.shape(),
                    },
                    prefill_attributes,
                );

                var o = prefill_output.o.splitAxis(.h, .{ .hkv = num_kv_heads, .hg = num_head_groups });

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

                const decode_output = Decode.paged_fa3_decode.call(
                    .{
                        .q = q_decode,
                        .paged_k = k_cache,
                        .paged_v = v_cache,
                        .cu_seqlens_q = cu_seqlens_q_decode,
                        .seqused_k = seqused_k_decode,
                        .block_table = block_table_decode,
                        .softmax_lse = softmax_lse_decode,
                        .softmax_lse_accum = softmax_lse_accum_decode,
                        .out_accum = out_accum,
                        .scheduler_metadata = scheduler_metadata_decode,
                    },
                    .{
                        .o = q_decode.shape(),
                    },
                    decode_attributes,
                );
                const o_decode = decode_output.o.splitAxis(.h, .{ .hkv = num_kv_heads, .hg = num_head_groups });

                o = o.dynamicUpdateSlice1d(o_decode, 0, context.decode_offset.?);
                break :b o;
            },
        };

        return o;
    }
};
