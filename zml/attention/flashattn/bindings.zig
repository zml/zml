const std = @import("std");

const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const runfiles = @import("runfiles");

const zml = @import("../../zml.zig");
const ffi = zml.pjrt.ffi;

fn flashattnDataTypeFromZmlDataType(dtype: zml.DataType) c.DataType {
    return switch (dtype) {
        .f32 => c.CAPI_FLOAT,
        .f16 => c.CAPI_FLOAT16,
        .bf16 => c.CAPI_BFLOAT16,
        .i32 => c.CAPI_INT32,
        .i8 => c.CAPI_INT8,
        else => unreachable,
    };
}

pub fn tensorFromBuffer(buffer: zml.Buffer, platform: zml.Platform) !c.FlashattnTensor {
    const rank: usize = buffer.rank();
    var dims: [8]i64 = undefined;
    @memcpy(dims[0..rank], buffer.shape().dims());
    var strides: [8]i64 = undefined;
    @memcpy(strides[0..rank], buffer.shape().withDtype(.u8).computeStrides().constSlice());

    const tensor: c.FlashattnTensor = .{
        .ptr = try buffer._shards.get(0).getOpaqueDeviceMemoryDataPointer(platform.pjrt_api),
        .rank = buffer.rank(),
        .dims = dims,
        .strides = strides,
        .dtype = flashattnDataTypeFromZmlDataType(buffer.dtype()),
    };

    return tensor;
}

pub const Buffer = struct {
    shape: zml.Shape,
    ptr: ?*anyopaque,

    pub fn toFlashattnTensor(buffer: Buffer) c.FlashattnTensor {
        const rank: usize = buffer.shape.rank();
        var dims: [8]i64 = undefined;
        @memcpy(dims[0..rank], buffer.shape.dims());
        var strides: [8]i64 = undefined;
        @memcpy(strides[0..rank], buffer.shape.withDtype(.u8).computeByteStrides().constSlice());

        const tensor: c.FlashattnTensor = .{
            .ptr = buffer.ptr,
            .rank = buffer.shape.rank(),
            .dims = dims,
            .strides = strides,
            .dtype = flashattnDataTypeFromZmlDataType(buffer.shape.dtype()),
        };

        return tensor;
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
    const dtype = dataTypeFromFfiDataType(buffer.dtype);
    return zml.Shape.init(buffer.dims(), dtype);
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
        pub fn register(platform: zml.Platform) !void {
            try platform.pjrt_api.ffi().?.register(platform.pjrt_api, T.custom_call_name, "cuda", T.run, .{ .command_buffer_compatible = true });
        }

        pub fn run(call_frame: *ffi.CallFrame) callconv(.c) ?*ffi.Error {
            return @field(T, @tagName(run_func))(call_frame) catch b: {
                break :b ffi.Error.create(call_frame.api.?, .unknown, "Unknown");
            };
        }
    };
}

pub const Fa2 = struct {
    pub const custom_call_name = "fa2_mha_fwd";
    const Wrapped = Wrapper(@This(), .runInner);

    pub const register = Wrapped.register;
    pub const run = Wrapped.run;

    pub fn runInner(call_frame: *ffi.CallFrame) !?*ffi.Error {
        if (call_frame.registeringHook()) return null;

        const q = bufferFromFfiBuffer(call_frame.args.buffers()[0]);
        const k = bufferFromFfiBuffer(call_frame.args.buffers()[1]);
        const v = bufferFromFfiBuffer(call_frame.args.buffers()[2]);
        const softmax_lse = bufferFromFfiBuffer(call_frame.args.buffers()[3]);
        const softmax_lse_accum = bufferFromFfiBuffer(call_frame.args.buffers()[4]);
        const out_accum = bufferFromFfiBuffer(call_frame.args.buffers()[5]);
        const o = bufferFromFfiBuffer(call_frame.results.buffers()[0]);

        const softmax_scale: f32 = getScalarAttributeAs(f32, call_frame, "softmax_scale") orelse b: {
            const head_dim = q.shape.dim(3);
            break :b 1.0 / std.math.sqrt(@as(f32, @floatFromInt(head_dim)));
        };
        const is_causal: bool = getScalarAttributeAs(bool, call_frame, "is_causal").?;
        const window_size_left: i32 = getScalarAttributeAs(i32, call_frame, "window_size_left") orelse -1;
        const window_size_right: i32 = getScalarAttributeAs(i32, call_frame, "window_size_right") orelse -1;

        const ctx: *ffi.ExecutionContext = @constCast(call_frame.ctx);
        const stream = call_frame.api.stream(ctx);

        const params: c.FA2MhaFwdParams = .{
            .is_causal = is_causal,
            .softmax_scale = softmax_scale,
            .window_size_left = window_size_left,
            .window_size_right = window_size_right,
        };

        fa2_mha_fwd.?(
            &q.toFlashattnTensor(),
            &k.toFlashattnTensor(),
            &v.toFlashattnTensor(),
            &o.toFlashattnTensor(),
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

pub const Fa2Varlen = struct {
    pub const custom_call_name = "fa2_mha_varlen_fwd";
    const Wrapped = Wrapper(@This(), .runInner);

    pub const register = Wrapped.register;
    pub const run = Wrapped.run;

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

        const params: c.FA2MhaVarlenFwdParams = .{
            .max_seqlen_q = max_seqlen_q,
            .max_seqlen_k = max_seqlen_k,
            .is_causal = is_causal,
            .softmax_scale = softmax_scale,
            .window_size_left = window_size_left,
            .window_size_right = window_size_right,
            .num_splits = 0,
            .num_heads = num_heads,
        };

        fa2_mha_varlen_fwd.?(
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
};

pub const Fa3 = struct {
    pub const custom_call_name = "fa3_mha_fwd";
    const Wrapped = Wrapper(@This(), .runInner);

    pub const register = Wrapped.register;
    pub const run = Wrapped.run;

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

        const params: c.FA3MhaFwdParams = .{
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

        fa3_mha_fwd.?(
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
};

const Fa2MhaFwdFunc = *const fn (
    q: ?*const c.FlashattnTensor,
    k: ?*const c.FlashattnTensor,
    v: ?*const c.FlashattnTensor,
    out: ?*const c.FlashattnTensor,
    softmax_lse: ?*const c.FlashattnTensor,
    alibi_slopes_: ?*const c.FlashattnTensor,
    softmax_lse_accum: ?*const c.FlashattnTensor,
    out_accum: ?*const c.FlashattnTensor,
    params: ?*const c.FA2MhaFwdParams,
    stream: ?*anyopaque,
) callconv(.c) void;

const Fa2MhaVarlenFwdFunc = *const fn (
    q: ?*const c.FlashattnTensor,
    k: ?*const c.FlashattnTensor,
    v: ?*const c.FlashattnTensor,
    out: ?*const c.FlashattnTensor,
    cu_seqlens_q: ?*const c.FlashattnTensor,
    cu_seqlens_k: ?*const c.FlashattnTensor,
    seqused_k: ?*const c.FlashattnTensor,
    block_table: ?*const c.FlashattnTensor,
    softmax_lse: ?*const c.FlashattnTensor,
    alibi_slopes_: ?*const c.FlashattnTensor,
    softmax_lse_accum: ?*const c.FlashattnTensor,
    out_accum: ?*const c.FlashattnTensor,
    params: ?*const c.FA2MhaVarlenFwdParams,
    stream: ?*anyopaque,
) callconv(.c) void;

const Fa3MhaFwdFunc = *const fn (
    q: ?*const c.FlashattnTensor,
    k: ?*const c.FlashattnTensor,
    v: ?*const c.FlashattnTensor,
    out: ?*const c.FlashattnTensor,
    cu_seqlens_q: ?*const c.FlashattnTensor,
    cu_seqlens_k: ?*const c.FlashattnTensor,
    seqused_q: ?*const c.FlashattnTensor,
    seqused_k: ?*const c.FlashattnTensor,
    page_table: ?*const c.FlashattnTensor,
    q_descale: ?*const c.FlashattnTensor,
    k_descale: ?*const c.FlashattnTensor,
    v_descale: ?*const c.FlashattnTensor,
    softmax_lse: ?*const c.FlashattnTensor,
    softmax_lse_accum: ?*const c.FlashattnTensor,
    out_accum: ?*const c.FlashattnTensor,
    scheduler_metadata: ?*const c.FlashattnTensor,
    s_aux: ?*const c.FlashattnTensor,
    cp_tot_seqused_k: ?*const c.FlashattnTensor,
    params: ?*const c.FA3MhaFwdParams,
    stream: ?*anyopaque,
) callconv(.c) void;

var fa2_mha_fwd: ?Fa2MhaFwdFunc = null;
var fa2_mha_varlen_fwd: ?Fa2MhaVarlenFwdFunc = null;
var fa3_mha_fwd: ?Fa3MhaFwdFunc = null;

pub fn load(allocator: std.mem.Allocator, io: std.Io) !void {
    var r_ = (try runfiles.Runfiles.create(.{ .allocator = allocator, .io = io })).?;
    defer r_.deinit(allocator);

    const source_repo = bazel_builtin.current_repository;
    const r = r_.withSourceRepo(source_repo);
    var buffer: [std.posix.PATH_MAX]u8 = undefined;
    const library = (try r.rlocation("flashattn/flashattn/lib/libflashattn.so", &buffer)) orelse return error.NotFound;
    const path = try std.posix.toPosixPath(library);
    const handle = std.c.dlopen(&path, .{ .NOW = true, .GLOBAL = true }) orelse {
        std.log.err("Failed to open libflashattn.so", .{});
        return;
    };

    fa2_mha_varlen_fwd = @ptrCast(@alignCast(std.c.dlsym(handle, "fa2_mha_varlen_fwd") orelse return error.NotFound));
    fa2_mha_fwd = @ptrCast(std.c.dlsym(handle, "fa2_mha_fwd") orelse return error.NotFound);
    fa3_mha_fwd = @ptrCast(@alignCast(std.c.dlsym(handle, "fa3_mha_fwd") orelse return error.NotFound));
}

pub fn register(platform: zml.Platform) !void {
    try Fa2.register(platform);
    try Fa2Varlen.register(platform);
    try Fa3.register(platform);
}
