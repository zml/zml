const std = @import("std");

const flashattn = @import("platforms/cuda/flashattn");
const platforms = @import("platforms");
const stdx = @import("stdx");

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
        try fa3.register(platform);
        try fa2.fa2_mha_varlen_fwd.register(platform);
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
        .u32 => .i32,
        .i32 => .i32,
        .i8 => .i8,
        else => std.debug.panic("Unsupported dtype for attention backend cu_fa2: {t}", .{dtype}),
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
    const Input = struct {
        q: zml.Tensor,
        k: zml.Tensor,
        v: zml.Tensor,
        cu_seqlens_q: zml.Tensor,
        cu_seqlens_k: zml.Tensor,
        seqused_k: zml.Tensor,
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
        input: zml.pjrtx.TensorToCustomCallBuffer(Input),
        output: zml.pjrtx.ShapeToCustomCallBuffer(Output),
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
            &toFlashattnTensor(input.seqused_k),
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
        sliding_window: i32 = -1,

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

        pub fn initBuffer(self: Metadata, io: std.Io, platform: *const zml.Platform, sharding: zml.Sharding) !zml.Bufferized(Metadata) {
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

    pub fn attention(q_: zml.Tensor, k_: zml.Tensor, v_: zml.Tensor, token_index_: zml.Tensor, metadata: Metadata, parameters: Parameters) zml.Tensor {
        const ctx = zml.Compiler.current();

        var bs: i64 = 1;
        var q = q_;
        var k = k_;
        var v = v_;
        var token_index = token_index_;

        // Follow cu_fa2 convention
        // We are a bit more restrictive because we assume each sequence has the same number of queries and keys.
        // https://deepwiki.com/Dao-AILab/flash-attention/3.1-flashattention-2-python-interface#flash_attn_varlen_func
        if (q.shape().hasTag(.b)) |_| {
            bs = q.dim(.b);
            q = q.merge(.{ .tot = .{ .b, .q } });
            k = k.merge(.{ .tot = .{ .b, .k } });
            v = v.merge(.{ .tot = .{ .b, .k } });
        } else {
            q = q.rename(.{ .q = .tot });
            k = k.rename(.{ .k = .tot });
            v = v.rename(.{ .k = .tot });
            token_index = token_index.broad(.init(.{ .b = 1 }, .i32));
        }

        const max_seqlen_q: i32 = @intCast(q_.dim(.q));
        const max_seqlen_k: i32 = @intCast(k_.dim(.k));

        // We still have a rectangle layout, q and k haven't been compacted.
        // So each sequence have the same number of queries, keys
        const cu_seqlens_q: zml.Tensor = .arange(.{ .end = max_seqlen_q * (bs + 1), .step = max_seqlen_q }, .i32);
        const cu_seqlens_k: zml.Tensor = .arange(.{ .end = max_seqlen_k * (bs + 1), .step = max_seqlen_k }, .i32);
        // But here we correctly pass the number of used k per sequence
        const seqused_k = token_index.addConstant(max_seqlen_q).convert(.i32);

        const original_tot = q.dim(.tot);
        const num_heads: i32 = @intCast(q_.dim(.h));
        const num_k_heads = k_.dim(.h);
        const head_dim = q.dim(.hd);
        const ngroups = @divExact(num_heads, num_k_heads);
        const seqlenq_ngroups_swapped = max_seqlen_q == 1 and num_heads > num_k_heads and @mod(head_dim, 8) == 0 and parameters.sliding_window < 0;
        if (seqlenq_ngroups_swapped) {
            q = q.splitAxis(.h, .{ .h = num_k_heads, .ngroups = ngroups })
                .transpose(.{ .tot, .ngroups, .h, .hd })
                .merge(.{ .tot = .{ .tot, .ngroups } });
        }

        const q_sharded = q.withPartitioning(.{ .h = .model });
        const model_partitions: i32 = @intCast(ctx.partitioning.numPartitionsForLogicalAxis(q_sharded.shape(), .model) catch std.debug.panic("cu_fa2 attention backend requires a .model sharding", .{}));

        const output = fa2_mha_varlen_fwd.call(
            .{
                .q = q_sharded,
                .k = k.withPartitioning(.{ .h = .model }),
                .v = v.withPartitioning(.{ .h = .model }),
                .cu_seqlens_q = cu_seqlens_q,
                .cu_seqlens_k = cu_seqlens_k,
                .seqused_k = seqused_k,
                .softmax_lse = metadata.softmax_lse.withPartitioning(.{ .h = .model }),
                .softmax_lse_accum = metadata.softmax_lse_accum.withPartitioning(.{ .h = .model }),
                .out_accum = metadata.out_accum.withPartitioning(.{ .h = .model }),
            },
            .{
                .o = q_sharded.shape(),
            },
            .{
                .softmax_scale = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(head_dim))),
                .is_causal = true,
                .window_size_left = parameters.sliding_window,
                .window_size_right = -1,
                .max_seqlen_q = max_seqlen_q,
                .max_seqlen_k = max_seqlen_k,
                .num_heads = @divExact(num_heads, model_partitions),
            },
        );
        var o = output.o;

        if (seqlenq_ngroups_swapped) {
            o = o.splitAxis(.tot, .{ .tot = original_tot, .ngroups = ngroups })
                .transpose(.{ .tot, .h, .ngroups, .hd })
                .merge(.{ .h = .{ .h, .ngroups } });
        }

        return if (q_.shape().hasTag(.b)) |_|
            o.splitAxis(.tot, .{ .b = bs, .q = q_.dim(.q) })
        else
            o.rename(.{ .tot = .q });
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

        pub fn initBuffer(self: Metadata, io: std.Io, platform: *const zml.Platform, sharding: zml.Sharding) !zml.Bufferized(Metadata) {
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
                .output_operand_aliases = &.{.{ .output_index = 0, .operand_index = 0 }},
                .has_side_effect = false,
            },
        );

        return o.splitAxis(.tot, .{ .b = 1, .q = q_.dim(.q) }).squeeze(.b);
    }
};

pub const paged_fa2 = struct {
    // God knows why flash attention uses this number and not something else.
    pub const MAX_NUM_SPLITS = 8;

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
    };

    pub const Prefill = struct {
        pub const custom_call_name = "paged_fa2_prefill";
        const Wrapped = Wrapper(@This(), .runInner);

        const register = Wrapped.register;
        const run = Wrapped.run;

        pub fn runInner(call_frame: *ffi.CallFrame) !?*ffi.Error {
            if (call_frame.registeringHook()) return null;

            const q = bufferFromFfiBuffer(call_frame.args.buffers()[0]);
            const paged_k = bufferFromFfiBuffer(call_frame.args.buffers()[1]);
            const paged_v = bufferFromFfiBuffer(call_frame.args.buffers()[2]);
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

            const q = bufferFromFfiBuffer(call_frame.args.buffers()[0]);
            const paged_k = bufferFromFfiBuffer(call_frame.args.buffers()[1]);
            const paged_v = bufferFromFfiBuffer(call_frame.args.buffers()[2]);
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

    pub fn pagedAttention(parameters: Parameters, q: zml.Tensor, k_cache: zml.Tensor, v_cache: zml.Tensor, opts: AttentionOptions) zml.Tensor {
        stdx.debug.assert(q.shape().hasTags(.{ .b, .hg, .hkv, .hd }), "Expected q to have tags .b, .h, .hd", .{});
        stdx.debug.assert(k_cache.shape().hasTags(.{ .page, .k_chunk, .hkv, .hd }), "Expected paged_k to have tags .page, .k_chunk, .h, .hd, got {}", .{k_cache.shape()});
        stdx.debug.assert(v_cache.shape().hasTags(.{ .page, .k_chunk, .hkv, .hd }), "Expected paged_v to have tags .page, .k_chunk, .h, .hd. got {}", .{v_cache.shape()});
        const ctx = zml.Compiler.current();
        const window_size_left = windowSizeLeft(opts.sliding_window);
        const max_seqlen_k: usize = @intCast(parameters.block_table.dim(.p) * k_cache.dim(.k_chunk));

        const num_head_groups = q.dim(.hg);
        const num_kv_heads = q.dim(.hkv);
        const head_dim = q.dim(.hd);
        const num_heads = num_head_groups * num_kv_heads;
        // FIXME: remove unreachable and propagate error correctly.
        const num_heads_per_shard = @divExact(num_heads, ctx.partitioning.numPartitionsForLogicalAxis(q.shape(), .model) catch unreachable);

        const o = if (parameters.options_.is_prefill) b: {
            const block_table = parameters.block_table.withPartitioning(.{ .b = .replicated });
            const query_offsets = splitQueryOffsets(parameters.query_start_len.withPartitioning(.{ .b = .replicated }));
            const cu_seqlens_q_prefill = query_offsets.prefill;
            const seqused_k = parameters.seq_lens.withPartitioning(.{ .b = .replicated });
            const cu_seqlens_q_decode = query_offsets.decode;

            const out_accum_prefill = zml.Tensor.uninitialized(.init(.{
                .splits = MAX_NUM_SPLITS,
                .hkv = num_kv_heads,
                .hg = num_head_groups,
                .b = q.dim(.b),
                .hd = head_dim,
            }, .f32)).withPartitioning(.{ .hkv = .model });
            const softmax_lse_prefill = zml.Tensor.uninitialized(.init(.{
                .hkv = num_kv_heads,
                .hg = num_head_groups,
                .b = q.dim(.b),
            }, .f32)).withPartitioning(.{ .hkv = .model });
            const softmax_lse_accum_prefill = zml.Tensor.uninitialized(.init(.{
                .splits = MAX_NUM_SPLITS,
                .hkv = num_kv_heads,
                .hg = num_head_groups,
                .b = q.dim(.b),
            }, .f32)).withPartitioning(.{ .hkv = .model });
            const dummy_cu_seqlens_k_prefill: zml.Tensor = .zeroes(cu_seqlens_q_prefill.shape());

            var q2 = q;
            q2 = q2.transpose(.{ .b, .hkv, .hg, .hd }).merge(.{ .h = .{ .hkv, .hg } }).withPartitioning(.{ .h = .model });

            const output_shape = q2.shape();
            const output_shapes: [4]zml.Shape = .{ output_shape, softmax_lse_prefill.shape(), softmax_lse_accum_prefill.shape(), out_accum_prefill.shape() };
            const prefill = zml.ops.manualComputation(
                (struct {
                    inputs: struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor },
                    metadata: struct {
                        is_causal: bool,
                        max_seqlen_k: usize,
                        max_seqlen_q: usize,
                        num_heads: i64,
                        window_size_left: i32,
                        window_size_right: i32,
                        softmax_scale: ?f32,
                    },
                    opts: zml.ops.CustomCallOptions,

                    fn body(self: @This(), output: []const zml.Shape) []const zml.Tensor {
                        return zml.ops.typedCustomCall(Prefill.custom_call_name, self.opts, self.inputs, output, self.metadata);
                    }
                }).body,
                .{
                    .inputs = .{
                        q2,
                        k_cache,
                        v_cache,
                        cu_seqlens_q_prefill,
                        dummy_cu_seqlens_k_prefill,
                        seqused_k,
                        block_table,
                        softmax_lse_prefill,
                        softmax_lse_accum_prefill,
                        out_accum_prefill,
                    },
                    .metadata = .{
                        .is_causal = opts.is_causal,
                        .max_seqlen_k = max_seqlen_k,
                        .max_seqlen_q = parameters.options_.max_seqlen_q,
                        .num_heads = num_heads_per_shard,
                        .window_size_left = window_size_left,
                        .window_size_right = if (opts.is_causal or parameters.options_.max_seqlen_q == 1) 0 else -1,
                        .softmax_scale = opts.scale,
                    },
                    .opts = zml.ops.CustomCallOptions{
                        .has_side_effect = false,
                        .output_operand_aliases = &.{
                            .{ .output_index = 1, .operand_index = 7 },
                            .{ .output_index = 2, .operand_index = 8 },
                            .{ .output_index = 3, .operand_index = 9 },
                        },
                    },
                },
                output_shapes,
            );

            // Decode consumes the prefill result and its scratch buffers in place.
            const dummy_cu_seqlens_k_decode = zml.Tensor.zeroes(cu_seqlens_q_decode.shape());
            const decode = zml.ops.manualComputation(
                (struct {
                    inputs: struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor },
                    metadata: struct {
                        is_causal: bool,
                        max_seqlen_k: usize,
                        num_heads: i64,
                        window_size_left: i32,
                        window_size_right: i32,
                        softmax_scale: ?f32,
                    },
                    opts: zml.ops.CustomCallOptions,

                    fn body(self: @This(), output: []const zml.Shape) []const zml.Tensor {
                        return zml.ops.typedCustomCall(Decode.custom_call_name, self.opts, self.inputs, output, self.metadata);
                    }
                }).body,
                .{
                    .inputs = .{
                        q2,
                        k_cache,
                        v_cache,
                        cu_seqlens_q_decode,
                        dummy_cu_seqlens_k_decode,
                        seqused_k,
                        block_table,
                        prefill[1],
                        prefill[2],
                        prefill[3],
                        prefill[0],
                    },
                    .metadata = .{
                        .is_causal = opts.is_causal,
                        .max_seqlen_k = max_seqlen_k,
                        .num_heads = num_heads_per_shard,
                        .window_size_left = window_size_left,
                        // FA2's GQA swap drops cu_seqlens_q. A zero right window
                        // preserves varlen rows and is equivalent for single-token decode.
                        .window_size_right = 0,
                        .softmax_scale = opts.scale,
                    },
                    .opts = zml.ops.CustomCallOptions{
                        .has_side_effect = false,
                        .output_operand_aliases = &.{
                            .{ .output_index = 0, .operand_index = 10 },
                            .{ .output_index = 1, .operand_index = 7 },
                            .{ .output_index = 2, .operand_index = 8 },
                            .{ .output_index = 3, .operand_index = 9 },
                        },
                    },
                },
                output_shapes,
            );

            break :b decode[0].splitAxis(.h, .{ .hkv = num_kv_heads, .hg = num_head_groups });
        } else b: {
            const seqlenq_ngroups_swapped = num_heads > num_kv_heads and @mod(head_dim, 8) == 0 and opts.sliding_window < 0;

            const block_table = parameters.block_table.withPartitioning(.{ .b = .replicated });
            const cu_seqlens_q = parameters.query_start_len.withPartitioning(.{ .b = .replicated });
            const seqused_k = parameters.seq_lens.withPartitioning(.{ .b = .replicated });

            const out_accum = zml.Tensor.uninitialized(.init(.{
                .splits = MAX_NUM_SPLITS,
                .hkv = num_kv_heads,
                .hg = num_head_groups,
                .b = q.dim(.b),
                .hd = head_dim,
            }, .f32)).withPartitioning(.{ .hkv = .model });

            const softmax_lse = zml.Tensor.uninitialized(.init(.{
                .hkv = num_kv_heads,
                .hg = num_head_groups,
                .b = q.dim(.b),
            }, .f32)).withPartitioning(.{ .hkv = .model });
            const softmax_lse_accum = zml.Tensor.uninitialized(.init(.{
                .splits = MAX_NUM_SPLITS,
                .hkv = num_kv_heads,
                .hg = num_head_groups,
                .b = q.dim(.b),
            }, .f32)).withPartitioning(.{ .hkv = .model });
            const dummy_cu_seqlens_k: zml.Tensor = .zeroes(cu_seqlens_q.shape());

            const batch_dim = q.dim(.b);
            var q2 = q;
            if (seqlenq_ngroups_swapped) {
                q2 = q2.transpose(.{ .b, .hg, .hkv, .hd }).merge(.{ .b = .{ .b, .hg } }).withPartitioning(.{ .hkv = .model });
            } else {
                q2 = q2.transpose(.{ .b, .hkv, .hg, .hd }).merge(.{ .h = .{ .hkv, .hg } }).withPartitioning(.{ .h = .model });
            }

            const output_shape = q2.shape();
            var o = zml.ops.manualComputation(
                (struct {
                    inputs: struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor },
                    metadata: struct {
                        is_causal: bool,
                        max_seqlen_k: usize,
                        num_heads: i64,
                        window_size_left: i32,
                        softmax_scale: ?f32,
                    },
                    opts: zml.ops.CustomCallOptions,

                    fn body(self: @This(), output: zml.Shape) zml.Tensor {
                        return zml.ops.customCall(Decode.custom_call_name, self.inputs, output, self.metadata, self.opts);
                    }
                }).body,
                .{
                    .inputs = .{
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
                    },
                    .metadata = .{
                        .is_causal = opts.is_causal,
                        .max_seqlen_k = max_seqlen_k,
                        .num_heads = num_heads_per_shard,
                        .window_size_left = window_size_left,
                        .softmax_scale = opts.scale,
                    },
                    .opts = zml.ops.CustomCallOptions{
                        .has_side_effect = false,
                    },
                },
                output_shape,
            );

            if (seqlenq_ngroups_swapped) {
                o = o.splitAxis(.b, .{ .b = batch_dim, .hg = num_head_groups }).transpose(.{ .b, .hkv, .hg, .hd });
            } else {
                o = o.splitAxis(.h, .{ .hkv = num_kv_heads, .hg = num_head_groups });
            }

            break :b o;
        };

        return o;
    }
};

pub const paged_fa3 = struct {
    // God knows why flash attention uses this number and not something else.
    pub const MAX_NUM_SPLITS = 32;

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
    };

    pub const Prefill = struct {
        pub const custom_call_name = "paged_fa3_prefill";
        const Wrapped = Wrapper(@This(), .runInner);

        const register = Wrapped.register;
        const run = Wrapped.run;

        pub fn runInner(call_frame: *ffi.CallFrame) !?*ffi.Error {
            if (call_frame.registeringHook()) return null;

            const q = bufferFromFfiBuffer(call_frame.args.buffers()[0]);
            const paged_k = bufferFromFfiBuffer(call_frame.args.buffers()[1]);
            const paged_v = bufferFromFfiBuffer(call_frame.args.buffers()[2]);
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
            const max_seqlen_q: i32 = getScalarAttributeAs(i32, call_frame, "max_seqlen_q").?;
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

            const q = bufferFromFfiBuffer(call_frame.args.buffers()[0]);
            const paged_k = bufferFromFfiBuffer(call_frame.args.buffers()[1]);
            const paged_v = bufferFromFfiBuffer(call_frame.args.buffers()[2]);
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

    pub fn pagedAttention(parameters: Parameters, q: zml.Tensor, k_cache: zml.Tensor, v_cache: zml.Tensor, opts: AttentionOptions) zml.Tensor {
        stdx.debug.assert(q.shape().hasTags(.{ .b, .hg, .hkv, .hd }), "Expected q to have tags .b, .h, .hd", .{});
        stdx.debug.assert(k_cache.shape().hasTags(.{ .page, .k_chunk, .hkv, .hd }), "Expected paged_k to have tags .page, .k_chunk, .h, .hd, got {}", .{k_cache.shape()});
        stdx.debug.assert(v_cache.shape().hasTags(.{ .page, .k_chunk, .hkv, .hd }), "Expected paged_v to have tags .page, .k_chunk, .h, .hd. got {}", .{v_cache.shape()});
        const window_size_left = windowSizeLeft(opts.sliding_window);
        const max_seqlen_k: usize = @intCast(parameters.block_table.dim(.p) * k_cache.dim(.k_chunk));

        const num_head_groups = q.dim(.hg);
        const num_kv_heads = q.dim(.hkv);
        const head_dim = q.dim(.hd);
        const o = if (parameters.options_.is_prefill) b: {
            const batch_size_prefill = parameters.block_table.dim(0);

            const block_table = parameters.block_table.withPartitioning(.{ .b = .replicated });
            const query_offsets = splitQueryOffsets(parameters.query_start_len.withPartitioning(.{ .b = .replicated }));
            const cu_seqlens_q_prefill = query_offsets.prefill;
            const seqused_k = parameters.seq_lens.withPartitioning(.{ .b = .replicated });
            const cu_seqlens_q_decode = query_offsets.decode;

            const out_accum_prefill = zml.Tensor.uninitialized(.init(.{
                .splits = MAX_NUM_SPLITS,
                .hkv = num_kv_heads,
                .hg = num_head_groups,
                .b = q.dim(.b),
                .hd = head_dim,
            }, .f32)).withPartitioning(.{ .hkv = .model });
            const softmax_lse_prefill = zml.Tensor.uninitialized(.init(.{
                .hkv = num_kv_heads,
                .hg = num_head_groups,
                .q = q.dim(.b),
            }, .f32)).withPartitioning(.{ .hkv = .model });
            const softmax_lse_accum_prefill = zml.Tensor.uninitialized(.init(.{
                .splits = MAX_NUM_SPLITS,
                .hkv = num_kv_heads,
                .hg = num_head_groups,
                .q = q.dim(.b),
            }, .f32)).withPartitioning(.{ .hkv = .model });
            const scheduler_metadata_prefill = zml.Tensor.zeroes(.init(.{ .b = batch_size_prefill + 1 }, .i32));

            const q2 = q.merge(.{ .h = .{ .hkv, .hg } }).withPartitioning(.{ .h = .model });

            const output_shape = q2.shape();
            const output_shapes: [4]zml.Shape = .{ output_shape, softmax_lse_prefill.shape(), softmax_lse_accum_prefill.shape(), out_accum_prefill.shape() };
            const prefill = zml.ops.manualComputation(
                (struct {
                    inputs: struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor },
                    metadata: struct {
                        is_causal: bool,
                        max_seqlen_k: usize,
                        max_seqlen_q: usize,
                        window_size_left: i32,
                    },
                    opts: zml.ops.CustomCallOptions,

                    fn body(self: @This(), output: []const zml.Shape) []const zml.Tensor {
                        return zml.ops.typedCustomCall(Prefill.custom_call_name, self.opts, self.inputs, output, self.metadata);
                    }
                }).body,
                .{
                    .inputs = .{
                        q2,
                        k_cache,
                        v_cache,
                        cu_seqlens_q_prefill,
                        seqused_k,
                        block_table,
                        softmax_lse_prefill,
                        softmax_lse_accum_prefill,
                        out_accum_prefill,
                        scheduler_metadata_prefill,
                    },
                    .metadata = .{
                        .is_causal = opts.is_causal,
                        .max_seqlen_k = max_seqlen_k,
                        .max_seqlen_q = parameters.options_.max_seqlen_q,
                        .window_size_left = window_size_left,
                    },
                    .opts = zml.ops.CustomCallOptions{
                        .has_side_effect = false,
                        .output_operand_aliases = &.{
                            .{ .output_index = 1, .operand_index = 6 },
                            .{ .output_index = 2, .operand_index = 7 },
                            .{ .output_index = 3, .operand_index = 8 },
                        },
                    },
                },
                output_shapes,
            );

            // Decode consumes the prefill result and its scratch buffers in place.
            const scheduler_metadata_decode = zml.Tensor.zeroes(scheduler_metadata_prefill.shape());
            const decode = zml.ops.manualComputation(
                (struct {
                    inputs: struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor },
                    metadata: struct {
                        is_causal: bool,
                        max_seqlen_k: usize,
                        window_size_left: i32,
                    },
                    opts: zml.ops.CustomCallOptions,

                    fn body(self: @This(), output: []const zml.Shape) []const zml.Tensor {
                        return zml.ops.typedCustomCall(Decode.custom_call_name, self.opts, self.inputs, output, self.metadata);
                    }
                }).body,
                .{
                    .inputs = .{
                        q2,
                        k_cache,
                        v_cache,
                        cu_seqlens_q_decode,
                        seqused_k,
                        block_table,
                        prefill[1],
                        prefill[2],
                        prefill[3],
                        scheduler_metadata_decode,
                        prefill[0],
                    },
                    .metadata = .{
                        .is_causal = opts.is_causal,
                        .max_seqlen_k = max_seqlen_k,
                        .window_size_left = window_size_left,
                    },
                    .opts = zml.ops.CustomCallOptions{
                        .has_side_effect = false,
                        .output_operand_aliases = &.{
                            .{ .output_index = 0, .operand_index = 10 },
                            .{ .output_index = 1, .operand_index = 6 },
                            .{ .output_index = 2, .operand_index = 7 },
                            .{ .output_index = 3, .operand_index = 8 },
                        },
                    },
                },
                output_shapes,
            );
            break :b decode[0].splitAxis(.h, .{ .hkv = num_kv_heads, .hg = num_head_groups });
        } else b: {
            const batch_size = parameters.block_table.dim(0);

            const block_table = parameters.block_table.withPartitioning(.{ .b = .replicated });
            const cu_seqlens_q = parameters.query_start_len.withPartitioning(.{ .b = .replicated });
            const seqused_k = parameters.seq_lens.withPartitioning(.{ .b = .replicated });

            const out_accum = zml.Tensor.uninitialized(.init(.{
                .splits = MAX_NUM_SPLITS,
                .hkv = num_kv_heads,
                .hg = num_head_groups,
                .b = q.dim(.b),
                .hd = head_dim,
            }, .f32)).withPartitioning(.{ .hkv = .model });
            const softmax_lse = zml.Tensor.uninitialized(.init(.{
                .hkv = num_kv_heads,
                .hg = num_head_groups,
                .q = q.dim(.b),
            }, .f32)).withPartitioning(.{ .hkv = .model });
            const softmax_lse_accum = zml.Tensor.uninitialized(.init(.{
                .splits = MAX_NUM_SPLITS,
                .q = q.dim(.b),
                .hkv = num_kv_heads,
                .hg = num_head_groups,
            }, .f32)).withPartitioning(.{ .hkv = .model });
            const scheduler_metadata = zml.Tensor.zeroes(.init(.{ .b = batch_size + 1 }, .i32));

            var q2 = q.merge(.{ .h = .{ .hkv, .hg } }).withPartitioning(.{ .h = .model });

            const output_shape = q2.shape();
            var o = zml.ops.manualComputation(
                (struct {
                    inputs: struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor },
                    metadata: struct {
                        is_causal: bool,
                        max_seqlen_k: usize,
                        window_size_left: i32,
                    },
                    opts: zml.ops.CustomCallOptions,

                    fn body(self: @This(), output: zml.Shape) zml.Tensor {
                        return zml.ops.customCall(Decode.custom_call_name, self.inputs, output, self.metadata, self.opts);
                    }
                }).body,
                .{
                    .inputs = .{
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
                    },
                    .metadata = .{
                        .is_causal = opts.is_causal,
                        .max_seqlen_k = max_seqlen_k,
                        .window_size_left = window_size_left,
                    },
                    .opts = zml.ops.CustomCallOptions{
                        .has_side_effect = false,
                    },
                },
                output_shape,
            );

            o = o.splitAxis(.h, .{ .hkv = num_kv_heads, .hg = num_head_groups });

            break :b o;
        };

        return o;
    }
};

fn windowSizeLeft(sliding_window: i32) i32 {
    return if (sliding_window > 0) sliding_window - 1 else sliding_window;
}

const QueryOffsets = struct {
    prefill: zml.Tensor,
    decode: zml.Tensor,
};

fn splitQueryOffsets(query_start_len: zml.Tensor) QueryOffsets {
    const starts = query_start_len.slice(.b, .{ .end = query_start_len.dim(.b) - 1 });
    const ends = query_start_len.slice(.b, .{ .start = 1 });
    // Prefill tokens are packed first. Any trailing single-token queries can use
    // decode, including one-token prefill chunks; empty rows stay empty in both calls.
    // Both kernels address the original Q and output buffers using absolute offsets.
    const decode_offset = ends.mask(ends.sub(starts).cmp(.GT, .scalar(1, .i32)), 0).max(.b).reshape(.{});
    return .{
        .prefill = query_start_len.minimum(decode_offset),
        .decode = query_start_len.maximum(decode_offset),
    };
}

test "FlashAttention sliding window uses an inclusive offset" {
    try std.testing.expectEqual(@as(i32, 2047), windowSizeLeft(2048));
    try std.testing.expectEqual(@as(i32, 0), windowSizeLeft(1));
    try std.testing.expectEqual(@as(i32, -1), windowSizeLeft(-1));
}
