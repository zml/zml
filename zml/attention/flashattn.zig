const std = @import("std");

const flashattn = @import("platforms/cuda/flashattn");
const platforms = @import("platforms");
const stdx = @import("stdx");

const zml = @import("../zml.zig");
const ffi = zml.pjrt.ffi;

pub fn load(allocator: std.mem.Allocator, io: std.Io) !void {
    if (comptime platforms.isEnabled(.cuda)) {
        try flashattn.load(allocator, io);
    }
}

pub fn register(platform: *zml.Platform) !void {
    if (comptime platforms.isEnabled(.cuda)) {
        try fa2.register(platform);
        try fa3.register(platform);
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

fn tensorFromBuffer(buffer: zml.Buffer, platform: *const zml.Platform) !flashattn.Tensor {
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
        pub fn register(platform: *zml.Platform) !void {
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
            // We must match the RANK of the Query tensor: [SeqLen, Heads, HeadDim]
            // Q is tagged as {.s, .h, .hd}.
            // We tag metadata similarly and shard on .h (the .model axis).

            return .{
                // LSE is [SeqLen, Heads]. We add a dummy 3rd dim to match Q's rank (3).
                .softmax_lse = .fromShape(zml.Shape.init(.{ opts.seqlen, opts.num_heads, 1 }, .f32)
                    .withTags(.{ .s, .h, .dummy })
                    .withPartitioning(.{ .s = .replicated, .h = .model, .dummy = .replicated })),

                // LSE Accum is used for split-K, usually [Heads, Splits]
                .softmax_lse_accum = .fromShape(zml.Shape.init(.{ 1, opts.num_heads, 128 }, .f32)
                    .withTags(.{ .dummy, .h, .hd })
                    .withPartitioning(.{ .dummy = .replicated, .h = .model, .hd = .replicated })),

                // Out Accum is [SeqLen, Heads, HeadDim]
                .out_accum = .fromShape(zml.Shape.init(.{ opts.seqlen, opts.num_heads, 128 }, .f32)
                    .withTags(.{ .s, .h, .hd })
                    .withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated })),
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

    pub fn attention(q_: zml.Tensor, k_: zml.Tensor, v_: zml.Tensor, token_index: zml.Tensor, metadata: Metadata, parameters: Parameters) zml.Tensor {
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
        const seqlenq_ngroups_swapped = max_seqlen_q == 1 and num_heads > num_heads_k and @mod(head_size, 8) == 0;
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
                .is_causal = true,
                .window_size_left = @as(i32, -1),
                .window_size_right = @as(i32, -1),
                .max_seqlen_q = max_seqlen_q,
                .max_seqlen_k = max_seqlen_k,
                .num_heads = @as(i32, @intCast(num_heads)),
            },
            .{
                .output_operand_aliases = &.{0},
                .has_side_effect = false,
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
                .scheduler_metadata = .init(.{2}, .i32),
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
                .has_side_effect = false,
            },
        );

        return o.splitAxis(.tot, .{ .b = 1, .q = q_.dim(.q) }).squeeze(.b);
    }
};
