const std = @import("std");

const dialects = @import("mlir/dialects");
const flashattn = @import("platforms/cuda/flashattn");
const mlir = @import("mlir");
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

    pub fn pagedAttention(parameters: Parameters, context: Context, q: zml.Tensor, k_cache: zml.Tensor, v_cache: zml.Tensor, layer_index_: zml.Tensor, opts: AttentionOptions) zml.Tensor {
        _ = layer_index_;
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

    pub fn pagedAttention(parameters: Parameters, context: Context, q: zml.Tensor, k_cache: zml.Tensor, v_cache: zml.Tensor, layer_index: zml.Tensor, opts: AttentionOptions) zml.Tensor {
        _ = layer_index;
        stdx.debug.assert(q.shape().hasTags(.{ .b, .hg, .hkv, .hd }), "Expected q to have tags .b, .h, .hd", .{});
        stdx.debug.assert(k_cache.shape().hasTags(.{ .page, .k_chunk, .hkv, .hd }), "Expected paged_k to have tags .page, .k_chunk, .h, .hd, got {}", .{k_cache.shape()});
        stdx.debug.assert(v_cache.shape().hasTags(.{ .page, .k_chunk, .hkv, .hd }), "Expected paged_v to have tags .page, .k_chunk, .h, .hd. got {}", .{v_cache.shape()});

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

pub const mosaic_tpu = struct {
    const RaggedPagedAttentionParams = struct {
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
        num_kv_pages_per_block: u32,
        num_queries_per_block: u32,
    };

    fn kernelFrontendAttrs(comp_ctx: anytype) *const mlir.Attribute {
        return mlir.dictionaryAttribute(comp_ctx.mlir_ctx, &.{
            mlir.NamedAttribute.named(comp_ctx.mlir_ctx, "kernel_metadata", mlir.stringAttribute(comp_ctx.mlir_ctx, "{}")),
        });
    }

    fn raggedPagedKernelCall(
        q: zml.Tensor,
        kv_pages: zml.Tensor,
        kv_lens: zml.Tensor,
        page_indices: zml.Tensor,
        cu_q_lens: zml.Tensor,
        num_seqs: zml.Tensor,
        backend_config: []const u8,
    ) zml.Tensor {
        const seq_buf_idx = zml.Tensor.constant(zml.DataType.i32.zero()).broad(.init(.{2}, .i32));

        const comp_ctx = zml.module.CompilationContext.current();
        const op = dialects.stablehlo.custom_call(
            comp_ctx.mlir_ctx,
            &.{
                kv_lens.value(),
                page_indices.value(),
                cu_q_lens.value(),
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

    pub const Options = struct {
        is_prefill: bool,
        batch_size: u32,
        max_num_pages: u32,
        max_seqlen_k: u32,
        max_token_count: u32,
        num_heads: u32,
        head_dim: u32,
        pages_per_compute_block: u32 = 1,

        pub fn isPrefill(self: @This()) bool {
            return self.is_prefill;
        }

        pub fn maxNumPages(self: @This()) usize {
            return self.max_num_pages;
        }
    };

    pub const Parameters = struct {
        opts: mosaic_tpu.Options,
        cu_seqlens_q: zml.Tensor,
        page_indices: zml.Tensor,
        kv_lens: zml.Tensor,

        pub fn init(opts: mosaic_tpu.Options) @This() {
            return .{
                .opts = opts,
                .cu_seqlens_q = zml.Tensor.init(.{ .n = @as(i64, opts.batch_size) + 1 }, .i32),
                .page_indices = zml.Tensor.init(.{ @as(i64, opts.batch_size), @as(i64, opts.max_num_pages) }, .i32),
                .kv_lens = zml.Tensor.init(.{@as(i64, opts.batch_size)}, .i32),
            };
        }

        pub fn options(self: @This()) mosaic_tpu.Options {
            return self.opts;
        }

        pub fn allocationSize(self: @This()) usize {
            var size: usize = 0;
            size += self.cu_seqlens_q.shape().byteSize();
            size += self.page_indices.shape().byteSize();
            size += self.kv_lens.shape().byteSize();
            return size;
        }
    };

    pub const Context = struct {
        max_num_pages: u32,
        pages_per_compute_block: u32,

        pub fn init(parameters: mosaic_tpu.Parameters) @This() {
            return .{
                .max_num_pages = parameters.opts.max_num_pages,
                .pages_per_compute_block = parameters.opts.pages_per_compute_block,
            };
        }
    };

    fn activeSequenceCount(cu_seqlens_q: zml.Tensor) zml.Tensor {
        const start = cu_seqlens_q.slice1d(0, .{ .end = cu_seqlens_q.dim(0) - 1 });
        const end = cu_seqlens_q.slice1d(0, .{ .start = 1 });
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

    fn renderBackendConfig(allocator: std.mem.Allocator, io: std.Io, params: RaggedPagedAttentionParams) []u8 {
        const json_string = std.fmt.allocPrint(
            allocator,
            "{{\"backend_config\":\"ragged_paged\",\"params\":{f}}}",
            .{std.json.fmt(params, .{})},
        ) catch |err| stdx.debug.panic("Failed to allocate ragged paged attention TPU params: {}", .{err});
        defer allocator.free(json_string);

        const compilation_context = zml.module.CompilationContext.current();
        const runtime = @constCast(compilation_context.platform).ensureTpuIrRuntime(compilation_context.allocator, io) catch |err| {
            stdx.debug.panic("Failed to initialize TPU IR runtime: {}", .{err});
        };
        return runtime.request(allocator, io, json_string) catch |err| {
            stdx.debug.panic("Failed to generate TPU backend config through persistent runtime: {}", .{err});
        };
    }

    const RaggedInputs = struct {
        q: zml.Tensor,
        kv_cache: zml.Tensor,
        kv_lens: zml.Tensor,
        page_indices: zml.Tensor,
        cu_seqlens_q: zml.Tensor,
        num_seqs: zml.Tensor,
    };

    inline fn shardRaggedInputs(
        parameters: Parameters,
        q: zml.Tensor,
        kv_cache: zml.Tensor,
    ) RaggedInputs {
        const q_sharded = q.withPartitioning(.{ .h = .model });
        const kv_cache_sharded = kv_cache.withPartitioning(.{ .hkv = .model });
        const cu_seqlens_q = parameters.cu_seqlens_q.withPartitioning(.{ .n = .replicated });

        return .{
            .q = q_sharded,
            .kv_cache = kv_cache_sharded,
            .kv_lens = parameters.kv_lens.withPartitioning(.{ ._0 = .replicated }),
            .page_indices = parameters.page_indices.withPartitioning(.{ ._0 = .replicated, ._1 = .replicated }),
            .cu_seqlens_q = cu_seqlens_q,
            .num_seqs = activeSequenceCount(cu_seqlens_q).withPartitioning(.{ ._0 = .replicated }),
        };
    }

    inline fn buildRaggedParams(
        inputs: RaggedInputs,
        parameters: Parameters,
        context: Context,
    ) RaggedPagedAttentionParams {
        const q_token_count = if (inputs.q.shape().hasTag(.q) != null) inputs.q.dim(.q) else inputs.q.dim(.b);
        const compilation_ctx = zml.module.CompilationContext.current();
        const model_partitions = compilation_ctx.partitioning.numPartitionsForLogicalAxis(inputs.q.shape(), .model) catch unreachable;
        stdx.debug.assert(model_partitions > 0, "Expected positive model partition count, got {}", .{model_partitions});

        const num_q_heads = inputs.q.dim(.h);
        const num_kv_heads = @divExact(inputs.kv_cache.dim(.hkv), 2);
        stdx.debug.assert(@mod(num_q_heads, model_partitions) == 0, "mosaic_tpu ragged paged attention expects q heads divisible by model partitions, got q_heads={} model_partitions={}", .{ num_q_heads, model_partitions });
        stdx.debug.assert(@mod(num_kv_heads, model_partitions) == 0, "mosaic_tpu ragged paged attention expects kv heads divisible by model partitions, got kv_heads={} model_partitions={}", .{ num_kv_heads, model_partitions });

        const max_num_seqs: i64 = @intCast(parameters.opts.batch_size);
        const num_queries_per_block: i64 = @max(@as(i64, 1), @min(q_token_count, max_num_seqs));
        return .{
            .num_q_tokens = q_token_count,
            .num_q_heads = @divExact(num_q_heads, model_partitions),
            .num_kv_heads = @divExact(num_kv_heads, model_partitions),
            .head_dim = inputs.q.dim(.hd),
            .q_dtype = jnpDTypeExpr(inputs.q.dtype()),
            .kv_dtype = jnpDTypeExpr(inputs.kv_cache.dtype()),
            .max_num_seqs = max_num_seqs,
            .pages_per_seq = context.max_num_pages,
            .total_num_pages = inputs.kv_cache.dim(.page),
            .page_size = inputs.kv_cache.dim(.k_chunk),
            .sm_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(parameters.opts.head_dim))),
            .num_kv_pages_per_block = context.pages_per_compute_block,
            .num_queries_per_block = @intCast(num_queries_per_block),
        };
    }

    const RaggedKernelBody = struct {
        fn body(ctx_: anytype, _: std.mem.Allocator, sharded_inputs: []const zml.Tensor, output: zml.Shape) zml.Tensor {
            stdx.debug.assert(sharded_inputs.len == 6, "mosaic_tpu ragged paged manualComputation expects 6 sharded inputs, got {}", .{sharded_inputs.len});
            const out = raggedPagedKernelCall(
                sharded_inputs[0],
                sharded_inputs[1],
                sharded_inputs[2],
                sharded_inputs[3],
                sharded_inputs[4],
                sharded_inputs[5],
                ctx_.backend_config,
            );
            stdx.debug.assert(out.shape().eql(output), "mosaic_tpu ragged paged manualComputation output shape mismatch, got {f}, expected {f}", .{ out.shape(), output });
            return out;
        }
    };

    pub fn raggedPagedAttention(parameters: @This().Parameters, context: @This().Context, q: zml.Tensor, kv_cache: zml.Tensor) zml.Tensor {
        stdx.debug.assert(
            q.shape().hasTags(.{ .b, .h, .hd }) or q.shape().hasTags(.{ .q, .h, .hd }),
            "mosaic_tpu ragged paged attention expects q to have tags (.b|.q, .h, .hd), got {f}",
            .{q.shape()},
        );
        stdx.debug.assert(
            kv_cache.shape().hasTags(.{ .page, .k_chunk, .hkv, .hd }),
            "mosaic_tpu ragged paged attention expects kv_cache to have tags (.page, .k_chunk, .hkv, .hd), got {f}",
            .{kv_cache.shape()},
        );

        var threaded: std.Io.Threaded = .init_single_threaded;
        threaded.allocator = zml.module.CompilationContext.current().allocator;
        defer threaded.deinit();

        const inputs = shardRaggedInputs(parameters, q, kv_cache);
        const backend_config = renderBackendConfig(
            threaded.allocator,
            threaded.io(),
            buildRaggedParams(inputs, parameters, context),
        );
        defer threaded.allocator.free(backend_config);

        return zml.ops.manualComputation(
            .{
                inputs.q,
                inputs.kv_cache,
                inputs.kv_lens,
                inputs.page_indices,
                inputs.cu_seqlens_q,
                inputs.num_seqs,
            },
            inputs.q.shape(),
            .{ .backend_config = backend_config },
            RaggedKernelBody.body,
        );
    }
};
