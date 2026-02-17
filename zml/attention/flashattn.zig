const std = @import("std");

const flashattn = @import("platforms/cuda/flashattn");
const platforms = @import("platforms");
const stdx = @import("stdx");

const zml = @import("../zml.zig");
const ffi = zml.pjrt.ffi;

const AttentionOptions = @import("paged_attention.zig").AttentionOptions;

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

fn tensorFromBuffer(buffer: zml.Buffer, platform: zml.Platform) !flashattn.Tensor {
    return .init(
        try buffer._shards.get(0).getOpaqueDeviceMemoryDataPointer(platform.pjrt_api),
        buffer.shape().dims(),
        buffer.shape().withDtype(.u8).computeStrides().constSlice(),
        flashattnDataTypeFromZmlDataType(buffer.dtype()),
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
            try platform.pjrt_api.ffi().?.register(platform.pjrt_api, T.custom_call_name, "cuda", T.run, .{ .command_buffer_compatible = true });
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

        pub fn init(opts: InitOptions) fa2.Parameters {
            _ = opts; // autofix
            return .{};
        }
    };

    pub const Metadata = struct {
        softmax_lse: zml.Tensor,
        softmax_lse_accum: zml.Tensor,
        out_accum: zml.Tensor,

        pub const InitOptions = struct {
            seqlen: i64,
        };

        pub fn init(opts: InitOptions) Metadata {
            return .{
                .softmax_lse = .init(.{1 * 32 * opts.seqlen * 4}, .i8),
                .softmax_lse_accum = .init(.{32 * 1 * 32 * 4 * 4}, .i8),
                .out_accum = .init(.{32 * 1 * 32 * 4 * 128 * 4}, .i8),
            };
        }

        pub fn initBuffer(self: Metadata, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(Metadata) {
            return .{
                .softmax_lse = try zml.Buffer.uninitialized(io, platform, self.softmax_lse.shape(), .{}),
                .softmax_lse_accum = try zml.Buffer.uninitialized(io, platform, self.softmax_lse_accum.shape(), .{}),
                .out_accum = try zml.Buffer.uninitialized(io, platform, self.out_accum.shape(), .{}),
            };
        }

        pub fn deinitBuffer(self: *zml.Bufferized(Metadata)) void {
            self.softmax_lse.deinit();
            self.softmax_lse_accum.deinit();
            self.out_accum.deinit();
        }
    };

    pub fn attention(q_: zml.Tensor, k_: zml.Tensor, v_: zml.Tensor, token_index: zml.Tensor, metadata: Metadata, parameters: Parameters, opts: AttentionOptions) zml.Tensor {
        _ = parameters; // autofix
        stdx.debug.assert(q_.shape().hasTag(.b) == null or q_.dim(.b) == 1, "fa2.attention support for batch size != 1 is not supported yet.", .{});
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
        const cu_seqlens_q = zml.Tensor.constantTensor(zml.Shape.init(.{2}, .i32), std.mem.sliceAsBytes(&[2]i32{ 0, max_seqlen_q }));

        const original_tot = q.dim(.tot);
        const num_heads = q_.dim(.h);
        const num_heads_k = k_.dim(.h);
        const head_size = q_.dim(.hd);
        const ngroups = @divExact(num_heads, num_heads_k);
        const seqlenq_ngroups_swapped = max_seqlen_q == 1 and num_heads > num_heads_k and @mod(head_size, 8) == 0 and opts.sliding_window < 0;
        if (seqlenq_ngroups_swapped) {
            q = q.splitAxis(.h, .{ .h = num_heads_k, .ngroups = ngroups }).transpose(.{ .tot, .ngroups, .h, .hd }).merge(.{ .tot = .{ .tot, .ngroups } });
        }

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
            },
            .{q.shape()},
            .{
                //.softmax_scale = @as(f32, 1.0),
                .is_causal = opts.is_causal,
                .window_size_left = opts.sliding_window,
                .window_size_right = @as(i32, -1),
                .max_seqlen_q = max_seqlen_q,
                .max_seqlen_k = max_seqlen_k,
                .num_heads = @as(i32, @intCast(num_heads)),
            },
            .{
                .output_operand_aliases = &.{0},
                .has_side_effect = true,
            },
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
            _ = opts; // autofix
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
        };

        pub fn init(opts: InitOptions) Metadata {
            return .{
                .softmax_lse = .init(.{1 * 32 * opts.seqlen * 4}, .i8),
                .softmax_lse_accum = .init(.{32 * 1 * 32 * 4 * 4}, .i8),
                .out_accum = .init(.{32 * 1 * 32 * 4 * 128 * 4}, .i8),
                .scheduler_metadata = .init(.{2}, .i32),
            };
        }

        pub fn initBuffer(self: Metadata, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(Metadata) {
            return .{
                .softmax_lse = try zml.Buffer.uninitialized(io, platform, self.softmax_lse.shape(), .{}),
                .softmax_lse_accum = try zml.Buffer.uninitialized(io, platform, self.softmax_lse_accum.shape(), .{}),
                .out_accum = try zml.Buffer.uninitialized(io, platform, self.out_accum.shape(), .{}),
                .scheduler_metadata = try zml.Buffer.uninitialized(io, platform, self.scheduler_metadata.shape(), .{}),
            };
        }

        pub fn deinitBuffer(self: *zml.Bufferized(Metadata)) void {
            self.softmax_lse.deinit();
            self.softmax_lse_accum.deinit();
            self.out_accum.deinit();
            self.scheduler_metadata.deinit();
        }
    };

    pub fn attention(q_: zml.Tensor, k_: zml.Tensor, v_: zml.Tensor, token_index: zml.Tensor, metadata: Metadata, parameters: Parameters, opts: AttentionOptions) zml.Tensor {
        _ = parameters; // autofix
        _ = opts; // autofix
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
        const cu_seqlens_q = zml.Tensor.constantTensor(zml.Shape.init(.{2}, .i32), std.mem.sliceAsBytes(&[2]i32{ 0, max_seqlen_q }));

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
                .has_side_effect = true,
            },
        );

        return o.splitAxis(.tot, .{ .b = 1, .q = q_.dim(.q) }).squeeze(.b);
    }
};

pub const paged_fa2 = struct {
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
                .block_table = .init(.{ options.batch_size, options.max_num_pages }, .i32),
                .cu_seqlens_q = .init(.{options.batch_size + 1}, .i32),
                .seqused_k = .init(.{options.batch_size}, .i32),
                .metadata = DecodeMetadata.init(options.max_token_count, options.num_heads, options.head_dim),
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

        pub fn init(max_token_count: usize, num_heads: usize, head_dim: usize) DecodeMetadata {
            return .{
                .out_accum = .init(.{8 * max_token_count * num_heads * head_dim * 4}, .i8),
            };
        }

        pub fn initBuffers(self: *const DecodeMetadata, io: std.Io, platform: zml.Platform) !zml.Bufferized(DecodeMetadata) {
            return .{
                .out_accum = try zml.Buffer.uninitialized(io, platform, self.out_accum.shape(), .{}),
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
                .block_table_prefill = .init(.{ options.batch_size_prefill, options.max_num_pages }, .i32),
                .cu_seqlens_q_prefill = .init(.{options.batch_size_prefill + 1}, .i32),
                .seqused_k_prefill = .init(.{options.batch_size_prefill}, .i32),
                .block_table_decode = .init(.{ options.batch_size_decode, options.max_num_pages }, .i32),
                .cu_seqlens_q_decode = .init(.{options.batch_size_decode + 1}, .i32),
                .seqused_k_decode = .init(.{options.batch_size_decode}, .i32),
                .metadata = MixedMetadata.init(options.max_token_count, options.num_heads, options.head_dim),
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

        pub fn init(max_token_count: usize, num_heads: usize, head_dim: usize) MixedMetadata {
            return .{
                .out_accum = .init(.{8 * max_token_count * num_heads * head_dim * 4}, .i8),
                .host_metadata = .init(.{2}, .i32),
            };
        }

        pub fn initBuffers(self: *const MixedMetadata, io: std.Io, platform: zml.Platform) !zml.Bufferized(MixedMetadata) {
            return .{
                .out_accum = try zml.Buffer.uninitialized(io, platform, self.out_accum.shape(), .{}),
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
            _ = num_heads; // autofix
            _ = num_kv_heads; // autofix
            _ = head_dim; // autofix
            _ = page_size; // autofix
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
                .num_splits = 0,
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
                .num_splits = 0,
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

    pub fn pagedAttention(parameters: Parameters, context: Context, q: zml.Tensor, k_cache: zml.Tensor, v_cache: zml.Tensor, layer_index: zml.Tensor, opts: AttentionOptions) zml.Tensor {
        stdx.debug.assert(q.shape().hasTags(.{ .b, .h, .hd }), "Expected q to have tags .b, .h, .hd", .{});
        stdx.debug.assert(k_cache.shape().hasTags(.{ .page, .k_chunk, .h, .hd }), "Expected paged_k to have tags .page, .k_chunk, .h, .hd, got {}", .{k_cache.shape()});
        stdx.debug.assert(v_cache.shape().hasTags(.{ .page, .k_chunk, .h, .hd }), "Expected paged_v to have tags .page, .k_chunk, .h, .hd. got {}", .{v_cache.shape()});

        const o = switch (parameters) {
            .decode => |decode_parameters| b: {
                const softmax_lse = zml.Tensor.constant(zml.DataType.f32.zero()).broad(.init(.{ .h = q.dim(.h), .q = q.dim(.b) }, .f32));
                const softmax_lse_accum = zml.Tensor.constant(zml.DataType.i8.zero()).broad(.init(.{64 * decode_parameters.block_table.dim(0) * q.dim(.h) * 4}, .i8));
                const dummy_cu_seqlens_k = zml.Tensor.constant(zml.DataType.i32.zero()).broad(decode_parameters.cu_seqlens_q.shape());

                const original_tot = q.dim(.b);
                const num_heads = q.dim(.h);
                const num_heads_k = k_cache.dim(.h);
                const head_size = q.dim(.hd);
                const ngroups = @divExact(num_heads, num_heads_k);
                const seqlenq_ngroups_swapped = num_heads > num_heads_k and @mod(head_size, 8) == 0 and opts.sliding_window < 0;
                var q2 = q;
                if (seqlenq_ngroups_swapped) {
                    q2 = q2.splitAxis(.h, .{ .h = num_heads_k, .ngroups = ngroups }).transpose(.{ .b, .ngroups, .h, .hd }).merge(.{ .b = .{ .b, .ngroups } });
                }

                var o = zml.ops.customCall(
                    Decode.custom_call_name,
                    .{
                        q2,
                        k_cache,
                        v_cache,
                        decode_parameters.cu_seqlens_q,
                        dummy_cu_seqlens_k,
                        decode_parameters.seqused_k,
                        decode_parameters.block_table,
                        softmax_lse,
                        softmax_lse_accum,
                        decode_parameters.metadata.out_accum,
                        layer_index,
                    },
                    .{q2.shape()},
                    .{
                        .is_causal = opts.is_causal,
                        .max_seqlen_k = context.max_seqlen_k,
                        .num_heads = num_heads,
                        .window_size_left = opts.sliding_window,
                    },
                    .{ .has_side_effect = false },
                );

                if (seqlenq_ngroups_swapped) {
                    o = o.splitAxis(.b, .{ .b = original_tot, .ngroups = ngroups }).transpose(.{ .b, .h, .ngroups, .hd }).merge(.{ .h = .{ .h, .ngroups } });
                }

                break :b o;
            },
            .mixed => |mixed_parameters| b: {
                const softmax_lse_prefill = zml.Tensor.constant(zml.DataType.f32.zero()).broad(.init(.{ .h = q.shape().dim(.h), .q = q.dim(.b) }, .f32));
                const softmax_lse_accum_prefill = zml.Tensor.constant(zml.DataType.i8.zero()).broad(.init(.{64 * q.dim(.b) * q.dim(.h) * 4}, .i8));
                const dummy_cu_seqlens_k_prefill = zml.Tensor.constant(zml.DataType.i32.zero()).broad(mixed_parameters.cu_seqlens_q_prefill.shape());
                var o = zml.ops.customCall(
                    Prefill.custom_call_name,
                    .{
                        q,
                        k_cache,
                        v_cache,
                        mixed_parameters.cu_seqlens_q_prefill,
                        dummy_cu_seqlens_k_prefill,
                        mixed_parameters.seqused_k_prefill,
                        mixed_parameters.block_table_prefill,
                        softmax_lse_prefill,
                        softmax_lse_accum_prefill,
                        mixed_parameters.metadata.out_accum,
                        mixed_parameters.metadata.host_metadata,
                        layer_index,
                    },
                    .{q.shape()},
                    .{
                        .is_causal = opts.is_causal,
                        .max_seqlen_k = context.max_seqlen_k,
                        .num_heads = q.dim(.h),
                        .window_size_left = opts.sliding_window,
                    },
                    .{ .has_side_effect = false },
                );

                const batch_size_decode = mixed_parameters.block_table_prefill.dim(0);
                const softmax_lse_decode = zml.Tensor.constant(zml.DataType.f32.zero()).broad(.init(.{ .h = q.shape().dim(.h), .q = batch_size_decode }, .f32));
                const softmax_lse_accum_decode = zml.Tensor.constant(zml.DataType.i8.zero()).broad(.init(.{64 * mixed_parameters.block_table_decode.dim(0) * q.dim(.h) * 4}, .i8));
                const dummy_cu_seqlens_k_decode = zml.Tensor.constant(zml.DataType.i32.zero()).broad(mixed_parameters.cu_seqlens_q_decode.shape());
                var q_decode = q.dynamicSlice1d(0, .{ .start = context.decode_offset.?, .len = batch_size_decode });

                const original_tot = q_decode.dim(.b);
                const num_heads = q_decode.dim(.h);
                const num_heads_k = k_cache.dim(.h);
                const head_size = q_decode.dim(.hd);
                const ngroups = @divExact(num_heads, num_heads_k);
                const seqlenq_ngroups_swapped = num_heads > num_heads_k and @mod(head_size, 8) == 0 and opts.sliding_window < 0;
                if (seqlenq_ngroups_swapped) {
                    q_decode = q_decode.splitAxis(.h, .{ .h = num_heads_k, .ngroups = ngroups }).transpose(.{ .b, .ngroups, .h, .hd }).merge(.{ .b = .{ .b, .ngroups } });
                }

                var o_decode = zml.ops.customCall(
                    Decode.custom_call_name,
                    .{
                        q_decode,
                        k_cache,
                        v_cache,
                        mixed_parameters.cu_seqlens_q_decode,
                        dummy_cu_seqlens_k_decode,
                        mixed_parameters.seqused_k_decode,
                        mixed_parameters.block_table_decode,
                        softmax_lse_decode,
                        softmax_lse_accum_decode,
                        mixed_parameters.metadata.out_accum,
                        layer_index,
                    },
                    .{q_decode.shape()},
                    .{
                        .is_causal = opts.is_causal,
                        .max_seqlen_k = context.max_seqlen_k,
                        .num_heads = q.dim(.h),
                        .window_size_left = opts.sliding_window,
                    },
                    .{ .has_side_effect = false },
                );

                if (seqlenq_ngroups_swapped) {
                    o_decode = o_decode.splitAxis(.b, .{ .b = original_tot, .ngroups = ngroups }).transpose(.{ .b, .h, .ngroups, .hd }).merge(.{ .h = .{ .h, .ngroups } });
                }

                o = o.dynamicUpdateSlice1d(o_decode, 0, context.decode_offset.?);
                break :b o;
            },
        };

        return o;
    }
};

pub const paged_fa3 = struct {
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
                .block_table = .init(.{ options.batch_size, options.max_num_pages }, .i32),
                .cu_seqlens_q = .init(.{options.batch_size + 1}, .i32),
                .seqused_k = .init(.{options.batch_size}, .i32),
                .metadata = DecodeMetadata.init(options.max_token_count, options.num_heads, options.head_dim),
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

        pub fn init(max_token_count: usize, num_heads: usize, head_dim: usize) DecodeMetadata {
            return .{
                .out_accum = .init(.{32 * max_token_count * num_heads * head_dim * 4}, .i8),
            };
        }

        pub fn initBuffers(self: *const DecodeMetadata, io: std.Io, platform: zml.Platform) !zml.Bufferized(DecodeMetadata) {
            return .{
                .out_accum = try zml.Buffer.uninitialized(io, platform, self.out_accum.shape(), .{}),
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
                .block_table_prefill = .init(.{ options.batch_size_prefill, options.max_num_pages }, .i32),
                .cu_seqlens_q_prefill = .init(.{options.batch_size_prefill + 1}, .i32),
                .seqused_k_prefill = .init(.{options.batch_size_prefill}, .i32),
                .block_table_decode = .init(.{ options.batch_size_decode, options.max_num_pages }, .i32),
                .cu_seqlens_q_decode = .init(.{options.batch_size_decode + 1}, .i32),
                .seqused_k_decode = .init(.{options.batch_size_decode}, .i32),
                .metadata = MixedMetadata.init(options.max_token_count, options.num_heads, options.head_dim),
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

        pub fn init(max_token_count: usize, num_heads: usize, head_dim: usize) MixedMetadata {
            return .{
                .out_accum = .init(.{32 * max_token_count * num_heads * head_dim * 4}, .i8),
                .host_metadata = .init(.{2}, .i32),
            };
        }

        pub fn initBuffers(self: *const MixedMetadata, io: std.Io, platform: zml.Platform) !zml.Bufferized(MixedMetadata) {
            return .{
                .out_accum = try zml.Buffer.uninitialized(io, platform, self.out_accum.shape(), .{}),
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
            _ = num_heads; // autofix
            _ = num_kv_heads; // autofix
            _ = head_dim; // autofix
            _ = page_size; // autofix
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
                .num_splits = 0,
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
                .num_splits = 0,
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
        _ = opts; // autofix
        stdx.debug.assert(q.shape().hasTags(.{ .b, .h, .hd }), "Expected q to have tags .b, .h, .hd", .{});
        stdx.debug.assert(k_cache.shape().hasTags(.{ .page, .k_chunk, .h, .hd }), "Expected paged_k to have tags .page, .k_chunk, .h, .hd, got {}", .{k_cache.shape()});
        stdx.debug.assert(v_cache.shape().hasTags(.{ .page, .k_chunk, .h, .hd }), "Expected paged_v to have tags .page, .k_chunk, .h, .hd. got {}", .{v_cache.shape()});

        const o = switch (parameters) {
            .decode => |decode_parameters| b: {
                const softmax_lse = zml.Tensor.constant(zml.DataType.f32.zero()).broad(.init(.{ .h = q.dim(.h), .q = q.dim(.b) }, .f32));
                const softmax_lse_accum = zml.Tensor.constant(zml.DataType.i8.zero()).broad(.init(.{32 * decode_parameters.block_table.dim(0) * q.dim(.h) * 4}, .i8));
                const scheduler_metadata = zml.Tensor.constant(zml.DataType.i32.zero()).broad(.init(.{ .b = decode_parameters.block_table.dim(0) + 1 }, .i32));

                const o = zml.ops.customCall(
                    Decode.custom_call_name,
                    .{
                        q,
                        k_cache,
                        v_cache,
                        decode_parameters.cu_seqlens_q,
                        decode_parameters.seqused_k,
                        decode_parameters.block_table,
                        softmax_lse,
                        softmax_lse_accum,
                        decode_parameters.metadata.out_accum,
                        scheduler_metadata,
                        layer_index,
                    },
                    .{q.shape()},
                    .{
                        .is_causal = true,
                        .max_seqlen_k = context.max_seqlen_k,
                    },
                    .{ .has_side_effect = false },
                );

                break :b o;
            },
            .mixed => |mixed_parameters| b: {
                const softmax_lse_prefill = zml.Tensor.constant(zml.DataType.f32.zero()).broad(.init(.{ .h = q.shape().dim(.h), .q = q.dim(.b) }, .f32));
                const softmax_lse_accum_prefill = zml.Tensor.constant(zml.DataType.i8.zero()).broad(.init(.{32 * q.dim(.b) * q.dim(.h) * 4}, .i8));
                const scheduler_metadata_prefill = zml.Tensor.constant(zml.DataType.i32.zero()).broad(.init(.{ .b = mixed_parameters.block_table_prefill.dim(0) + 1 }, .i32));
                var o = zml.ops.customCall(
                    Prefill.custom_call_name,
                    .{
                        q,
                        k_cache,
                        v_cache,
                        mixed_parameters.cu_seqlens_q_prefill,
                        mixed_parameters.seqused_k_prefill,
                        mixed_parameters.block_table_prefill,
                        softmax_lse_prefill,
                        softmax_lse_accum_prefill,
                        mixed_parameters.metadata.out_accum,
                        scheduler_metadata_prefill,
                        mixed_parameters.metadata.host_metadata,
                        layer_index,
                    },
                    .{q.shape()},
                    .{
                        .is_causal = false,
                        .max_seqlen_k = context.max_seqlen_k,
                    },
                    .{ .has_side_effect = false },
                );

                const batch_size_decode = mixed_parameters.block_table_prefill.dim(0);
                const softmax_lse_decode = zml.Tensor.constant(zml.DataType.f32.zero()).broad(.init(.{ .h = q.shape().dim(.h), .q = batch_size_decode }, .f32));
                const softmax_lse_accum_decode = zml.Tensor.constant(zml.DataType.i8.zero()).broad(.init(.{32 * mixed_parameters.block_table_decode.dim(0) * q.dim(.h) * 4}, .i8));
                const scheduler_metadata_decode = zml.Tensor.constant(zml.DataType.i32.zero()).broad(.init(.{ .b = mixed_parameters.block_table_decode.dim(0) + 1 }, .i32));
                var q_decode = q.dynamicSlice1d(0, .{ .start = context.decode_offset.?, .len = batch_size_decode });

                const o_decode = zml.ops.customCall(
                    Decode.custom_call_name,
                    .{
                        q_decode,
                        k_cache,
                        v_cache,
                        mixed_parameters.cu_seqlens_q_decode,
                        mixed_parameters.seqused_k_decode,
                        mixed_parameters.block_table_decode,
                        softmax_lse_decode,
                        softmax_lse_accum_decode,
                        mixed_parameters.metadata.out_accum,
                        scheduler_metadata_decode,
                        layer_index,
                    },
                    .{q_decode.shape()},
                    .{
                        .is_causal = true,
                        .max_seqlen_k = context.max_seqlen_k,
                    },
                    .{ .has_side_effect = false },
                );

                o = o.dynamicUpdateSlice1d(o_decode, 0, context.decode_offset.?);
                break :b o;
            },
        };

        return o;
    }
};
