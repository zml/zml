const std = @import("std");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
pub const FA2MhaFwdParams = c.FA2MhaFwdParams;
pub const FA2MhaVarlenFwdParams = c.FA2MhaVarlenFwdParams;
pub const FA3MhaFwdParams = c.FA3MhaFwdParams;
const runfiles = @import("runfiles");

pub const Tensor = struct {
    inner: c.FlashattnTensor,

    pub fn init(ptr: *anyopaque, dims: []const i64, strides: []const i64, dtype: DataType) Tensor {
        const rank = dims.len;
        var ret: Tensor = .{
            .inner = .{
                .ptr = ptr,
                .rank = @intCast(rank),
                .dims = undefined,
                .strides = undefined,
                .dtype = @intCast(@intFromEnum(dtype)),
            },
        };
        @memcpy(ret.inner.dims[0..rank], dims);
        @memcpy(ret.inner.strides[0..rank], strides);
        return ret;
    }
};

pub const DataType = enum(c_int) {
    f32 = c.CAPI_FLOAT,
    f16 = c.CAPI_FLOAT16,
    bf16 = c.CAPI_BFLOAT16,
    i32 = c.CAPI_INT32,
    i8 = c.CAPI_INT8,
};

const Fa2MhaFwdFunc = *const fn (
    q: *const Tensor,
    k: *const Tensor,
    v: *const Tensor,
    out: *const Tensor,
    softmax_lse: ?*const Tensor,
    alibi_slopes_: ?*const Tensor,
    softmax_lse_accum: ?*const Tensor,
    out_accum: ?*const Tensor,
    params: ?*const FA2MhaFwdParams,
    stream: ?*anyopaque,
) callconv(.c) void;

const Fa2MhaVarlenFwdFunc = *const fn (
    q: *const Tensor,
    k: *const Tensor,
    v: *const Tensor,
    out: *const Tensor,
    cu_seqlens_q: ?*const Tensor,
    cu_seqlens_k: ?*const Tensor,
    seqused_k: ?*const Tensor,
    block_table: ?*const Tensor,
    softmax_lse: ?*const Tensor,
    alibi_slopes_: ?*const Tensor,
    softmax_lse_accum: ?*const Tensor,
    out_accum: ?*const Tensor,
    params: ?*const FA2MhaVarlenFwdParams,
    stream: ?*anyopaque,
) callconv(.c) void;

const Fa3MhaFwdFunc = *const fn (
    q: ?*const Tensor,
    k: ?*const Tensor,
    v: ?*const Tensor,
    out: ?*const Tensor,
    cu_seqlens_q: ?*const Tensor,
    cu_seqlens_k: ?*const Tensor,
    seqused_q: ?*const Tensor,
    seqused_k: ?*const Tensor,
    page_table: ?*const Tensor,
    q_descale: ?*const Tensor,
    k_descale: ?*const Tensor,
    v_descale: ?*const Tensor,
    softmax_lse: ?*const Tensor,
    softmax_lse_accum: ?*const Tensor,
    out_accum: ?*const Tensor,
    scheduler_metadata: ?*const Tensor,
    s_aux: ?*const Tensor,
    cp_tot_seqused_k: ?*const Tensor,
    params: ?*const FA3MhaFwdParams,
    stream: ?*anyopaque,
) callconv(.c) void;

pub var fa2_mha_fwd: Fa2MhaFwdFunc = undefined;
pub var fa2_mha_varlen_fwd: Fa2MhaVarlenFwdFunc = undefined;
pub var fa3_mha_fwd: Fa3MhaFwdFunc = undefined;

pub fn load(allocator: std.mem.Allocator, io: std.Io) !void {
    _ = allocator; // autofix

    const r = try bazel.runfiles(io, bazel_builtin.current_repository);

    var buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const library = (try r.rlocation("flashattn/flashattn/lib/libflashattn.so", &buffer)) orelse return error.NotFound;
    var lib = std.DynLib.open(library) catch |err| {
        std.log.err("Failed to open libflashattn.so: {any}", .{err});
        return err;
    };

    fa2_mha_varlen_fwd = lib.lookup(Fa2MhaVarlenFwdFunc, "fa2_mha_varlen_fwd") orelse return error.NotFound;
    fa2_mha_fwd = lib.lookup(Fa2MhaFwdFunc, "fa2_mha_fwd") orelse return error.NotFound;
    fa3_mha_fwd = lib.lookup(Fa3MhaFwdFunc, "fa3_mha_fwd") orelse return error.NotFound;
}
