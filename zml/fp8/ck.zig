//! XLA FFI adapter for native ROCm Composable Kernel FP8 operations.

const std = @import("std");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");

const zml = @import("../zml.zig");

const GemmFn = *const fn (
    stream: ?*anyopaque,
    a: *const anyopaque,
    b: *const anyopaque,
    a_scale: *const f32,
    b_scale: *const f32,
    c: *anyopaque,
    m: i64,
    n: i64,
    k: i64,
) callconv(.c) c_int;

const MoeInitFn = *const fn (
    code_object_path: [*:0]const u8,
) callconv(.c) c_int;

const MoeFn = *const fn (
    stream: ?*anyopaque,
    sorted_token_ids: *const i32,
    sorted_expert_ids: *const i32,
    num_valid_ids: *const i32,
    sorted_weights: *const f32,
    hidden: *const anyopaque,
    gate_up: *const anyopaque,
    down: *const anyopaque,
    input_scale: *const f32,
    gate_up_scale: *const f32,
    down_scale: *const f32,
    output: *anyopaque,
    tokens: i64,
    experts: i64,
    topk: i64,
    sorted_blocks: i64,
    hidden_size: i64,
    intermediate_size: i64,
) callconv(.c) c_int;

const Api = struct {
    library: std.DynLib,
    gemm: GemmFn,
    moe: MoeFn,
};

var loaded_api: ?Api = null;

pub fn load(io: std.Io) !void {
    if (loaded_api != null) return;

    const runfiles = try bazel.runfiles(bazel_builtin.current_repository);
    var runfile_path_buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const runfile_path = (try runfiles.rlocation(
        "zml/platforms/rocm/ck_fp8/libzml_ck_fp8.so",
        &runfile_path_buffer,
    )) orelse return error.NotFound;

    var canonical_buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const canonical_length = if (std.fs.path.isAbsolute(runfile_path))
        try std.Io.Dir.realPathFileAbsolute(io, runfile_path, &canonical_buffer)
    else
        try std.Io.Dir.cwd().realPathFile(io, runfile_path, &canonical_buffer);

    var library = try std.DynLib.open(canonical_buffer[0..canonical_length]);
    errdefer library.close();
    const gemm_fn = library.lookup(GemmFn, "zml_ck_gemm_a8w8_blockscale") orelse
        return error.SymbolNotFound;
    const moe_init = library.lookup(MoeInitFn, "zml_aiter_moe_init") orelse
        return error.SymbolNotFound;
    const moe_fn = library.lookup(MoeFn, "zml_aiter_moe_a8w8_blockscale") orelse
        return error.SymbolNotFound;

    var code_object_path_buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const code_object_path = (try runfiles.rlocation(
        "zml/platforms/rocm/ck_fp8/fmoe_bf16_blockscale_fp8_gfx942.co",
        &code_object_path_buffer,
    )) orelse return error.NotFound;
    var code_object_path_z_buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const code_object_path_z = try std.fmt.bufPrintZ(&code_object_path_z_buffer, "{s}", .{code_object_path});
    if (moe_init(code_object_path_z.ptr) != 0) return error.CodeObjectLoadFailed;

    loaded_api = .{
        .library = library,
        .gemm = gemm_fn,
        .moe = moe_fn,
    };
}

const Input = struct {
    a: zml.Tensor,
    b: zml.Tensor,
    a_scale: zml.Tensor,
    b_scale: zml.Tensor,
};

const Output = struct {
    c: zml.Shape,
};

fn ffiCall(
    call_frame: *zml.pjrt.ffi.CallFrame,
    input: zml.pjrtx.TensorToCustomCallBuffer(Input),
    output: zml.pjrtx.ShapeToCustomCallBuffer(Output),
    _: void,
) !?*zml.pjrt.ffi.Error {
    const api = if (loaded_api) |*value| value else return error.BackendNotLoaded;
    if (input.a.shape.rank() != 2 or input.b.shape.rank() != 2 or
        input.a_scale.shape.rank() != 2 or input.b_scale.shape.rank() != 2 or
        output.c.shape.rank() != 2)
    {
        return error.InvalidShape;
    }
    if (input.a.shape.dtype() != .f8e4m3fnuz or input.b.shape.dtype() != .f8e4m3fnuz or
        input.a_scale.shape.dtype() != .f32 or input.b_scale.shape.dtype() != .f32 or
        output.c.shape.dtype() != .bf16)
    {
        return error.UnsupportedType;
    }

    const m = input.a.shape.dim(0);
    const k = input.a.shape.dim(1);
    const n = input.b.shape.dim(0);
    const k_blocks = std.math.divCeil(i64, k, 128) catch unreachable;
    const n_blocks = std.math.divCeil(i64, n, 128) catch unreachable;
    if (input.b.shape.dim(1) != k or
        input.a_scale.shape.dim(0) != m or input.a_scale.shape.dim(1) != k_blocks or
        input.b_scale.shape.dim(0) != n_blocks or input.b_scale.shape.dim(1) != k_blocks or
        output.c.shape.dim(0) != m or output.c.shape.dim(1) != n)
    {
        return error.InvalidShape;
    }

    const status = api.gemm(
        @ptrCast(call_frame.api.stream(call_frame.ctx)),
        input.a.ptr,
        input.b.ptr,
        @ptrCast(@alignCast(input.a_scale.ptr)),
        @ptrCast(@alignCast(input.b_scale.ptr)),
        output.c.ptr,
        m,
        n,
        k,
    );
    if (status != 0) return error.UnsupportedKernelArguments;
    return null;
}

const gemmCall = zml.ops.CustomCall(Input, Output, void, ffiCall, .{
    .name = "zml$ck_gemm_a8w8_blockscale",
    // The dense frontend places this call inside an explicit manual-computation
    // region, so wrapping it in another sharding-aware region would bind the
    // same mesh axis twice.
    .sharding_aware = false,
    .has_side_effect = false,
});

pub fn register(platform: *const zml.Platform) !void {
    try gemmCall.register(platform);
}

pub fn gemm(a: zml.Tensor, b: zml.Tensor, a_scale: zml.Tensor, b_scale: zml.Tensor, output: zml.Shape) zml.Tensor {
    return gemmCall.call(
        .{ .a = a, .b = b, .a_scale = a_scale, .b_scale = b_scale },
        .{ .c = output },
        {},
    ).c;
}

pub const MoeAttributes = struct {
    tokens: i64,
    experts: i64,
    topk: i64,
    sorted_blocks: i64,
    hidden_size: i64,
    intermediate_size: i64,
};

const MoeInput = struct {
    sorted_token_ids: zml.Tensor,
    sorted_expert_ids: zml.Tensor,
    num_valid_ids: zml.Tensor,
    sorted_weights: zml.Tensor,
    hidden: zml.Tensor,
    gate_up: zml.Tensor,
    down: zml.Tensor,
    input_scale: zml.Tensor,
    gate_up_scale: zml.Tensor,
    down_scale: zml.Tensor,
};

fn ffiMoe(
    call_frame: *zml.pjrt.ffi.CallFrame,
    input: zml.pjrtx.TensorToCustomCallBuffer(MoeInput),
    output: zml.pjrtx.ShapeToCustomCallBuffer(Output),
    attributes: MoeAttributes,
) !?*zml.pjrt.ffi.Error {
    const api = if (loaded_api) |*value| value else return error.BackendNotLoaded;
    if (input.sorted_token_ids.shape.dtype() != .i32 or input.sorted_expert_ids.shape.dtype() != .i32 or
        input.num_valid_ids.shape.dtype() != .i32 or input.sorted_weights.shape.dtype() != .f32 or
        input.hidden.shape.dtype() != .f8e4m3fnuz or input.gate_up.shape.dtype() != .f8e4m3fnuz or
        input.down.shape.dtype() != .f8e4m3fnuz or input.input_scale.shape.dtype() != .f32 or
        input.gate_up_scale.shape.dtype() != .f32 or input.down_scale.shape.dtype() != .f32 or
        output.c.shape.dtype() != .bf16)
    {
        return error.UnsupportedType;
    }
    if (input.sorted_token_ids.shape.rank() != 1 or input.sorted_expert_ids.shape.rank() != 1 or
        input.num_valid_ids.shape.rank() != 1 or input.sorted_weights.shape.rank() != 1 or
        input.hidden.shape.rank() != 2 or input.gate_up.shape.rank() != 3 or input.down.shape.rank() != 3 or
        input.input_scale.shape.rank() != 2 or input.gate_up_scale.shape.rank() != 3 or
        input.down_scale.shape.rank() != 3 or output.c.shape.rank() != 2)
    {
        return error.InvalidShape;
    }
    const k_blocks = @divExact(attributes.hidden_size, 128);
    const i_blocks = @divExact(attributes.intermediate_size, 128);
    if (input.sorted_token_ids.shape.dim(0) != input.sorted_weights.shape.dim(0) or
        input.sorted_expert_ids.shape.dim(0) != attributes.sorted_blocks or
        input.hidden.shape.dim(0) != attributes.tokens or input.hidden.shape.dim(1) != attributes.hidden_size or
        input.gate_up.shape.dim(0) != attributes.experts or input.gate_up.shape.dim(1) != 2 * attributes.intermediate_size or
        input.gate_up.shape.dim(2) != attributes.hidden_size or input.down.shape.dim(0) != attributes.experts or
        input.down.shape.dim(1) != attributes.hidden_size or input.down.shape.dim(2) != attributes.intermediate_size or
        input.input_scale.shape.dim(0) != k_blocks or input.input_scale.shape.dim(1) != attributes.tokens or
        input.gate_up_scale.shape.dim(0) != attributes.experts or input.gate_up_scale.shape.dim(1) != 2 * i_blocks or
        input.gate_up_scale.shape.dim(2) != k_blocks or input.down_scale.shape.dim(0) != attributes.experts or
        input.down_scale.shape.dim(1) != k_blocks or input.down_scale.shape.dim(2) != i_blocks or
        output.c.shape.dim(0) != attributes.tokens or output.c.shape.dim(1) != attributes.hidden_size)
    {
        return error.InvalidShape;
    }
    const status = api.moe(
        @ptrCast(call_frame.api.stream(call_frame.ctx)),
        @ptrCast(@alignCast(input.sorted_token_ids.ptr)),
        @ptrCast(@alignCast(input.sorted_expert_ids.ptr)),
        @ptrCast(@alignCast(input.num_valid_ids.ptr)),
        @ptrCast(@alignCast(input.sorted_weights.ptr)),
        input.hidden.ptr,
        input.gate_up.ptr,
        input.down.ptr,
        @ptrCast(@alignCast(input.input_scale.ptr)),
        @ptrCast(@alignCast(input.gate_up_scale.ptr)),
        @ptrCast(@alignCast(input.down_scale.ptr)),
        output.c.ptr,
        attributes.tokens,
        attributes.experts,
        attributes.topk,
        attributes.sorted_blocks,
        attributes.hidden_size,
        attributes.intermediate_size,
    );
    if (status != 0) return error.UnsupportedKernelArguments;
    return null;
}

const moeCall = zml.ops.CustomCall(MoeInput, Output, MoeAttributes, ffiMoe, .{
    .name = "zml$aiter_moe_a8w8_blockscale",
    // Expert sharding is handled by moe.forwardMoe's manual-computation region.
    .sharding_aware = false,
    .has_side_effect = false,
});

pub fn registerMoe(platform: *const zml.Platform) !void {
    try moeCall.register(platform);
}

pub fn moe(
    sorted_token_ids: zml.Tensor,
    sorted_expert_ids: zml.Tensor,
    num_valid_ids: zml.Tensor,
    sorted_weights: zml.Tensor,
    hidden: zml.Tensor,
    gate_up: zml.Tensor,
    down: zml.Tensor,
    input_scale: zml.Tensor,
    gate_up_scale: zml.Tensor,
    down_scale: zml.Tensor,
    output: zml.Shape,
    attributes: MoeAttributes,
) zml.Tensor {
    return moeCall.call(.{
        .sorted_token_ids = sorted_token_ids,
        .sorted_expert_ids = sorted_expert_ids,
        .num_valid_ids = num_valid_ids,
        .sorted_weights = sorted_weights,
        .hidden = hidden,
        .gate_up = gate_up,
        .down = down,
        .input_scale = input_scale,
        .gate_up_scale = gate_up_scale,
        .down_scale = down_scale,
    }, .{ .c = output }, attributes).c;
}

test {
    std.testing.refAllDecls(@This());
}
