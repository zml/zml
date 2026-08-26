const std = @import("std");

const zml = @import("zml");

const log = std.log.scoped(.minimax_h3_vision);

const Input = struct {
    x: zml.Tensor,
    w: zml.Tensor,
    bias: zml.Tensor,
};
const Output = struct { y: zml.Shape };
const Attrs = struct {};

/// Official `Qwen3VLVisionPatchEmbed` is cuDNN Conv3d. XLA Conv3d/Linear fold to the
/// same GEMM (1 bf16 ulp) and split merger row 1189.
const PatchEmbedCall = zml.ops.CustomCall(Input, Output, Attrs, patchEmbedFfi, .{
    .name = "h3$vision_patch_embed",
    .sharding_aware = false,
    .has_side_effect = false,
});

pub fn register(platform: *const zml.Platform) !void {
    try PatchEmbedCall.register(platform);
}

pub fn forward(proj: zml.nn.Linear, patches: zml.Tensor) zml.Tensor {
    const w = proj.weight;
    const bias = proj.bias orelse std.debug.panic("vision patch embed needs bias", .{});
    const seq = patches.dim(.s);
    const hidden = w.dim(.dout);
    const dt = w.dtype();
    const x = patches.squeeze(.b).reshape(.{
        .n = seq,
        .c = w.dim(.d),
        .t = w.dim(.kt),
        .h = w.dim(.kh),
        .w = w.dim(.kw),
    }).convert(dt);
    const y = PatchEmbedCall.call(
        .{ .x = x, .w = w, .bias = bias.convert(dt) },
        .{ .y = .init(.{ .n = seq, .d = hidden }, dt) },
        .{},
    );
    return y.y.reshape(.{ .b = 1, .s = seq, .d = hidden });
}

fn patchEmbedFfi(
    call_frame: *zml.pjrt.ffi.CallFrame,
    input: zml.pjrtx.TensorToCustomCallBuffer(Input),
    output: zml.pjrtx.ShapeToCustomCallBuffer(Output),
    _: Attrs,
) !?*zml.pjrt.ffi.Error {
    const stream = call_frame.stream() orelse return error.NoCudaStream;
    try cudnnPatchEmbed(@ptrCast(@constCast(stream)), input.x, input.w, input.bias, output.y);
    return null;
}

const Cudnn = struct {
    create: *const fn (*?*anyopaque) callconv(.c) c_int,
    set_stream: *const fn (?*anyopaque, ?*anyopaque) callconv(.c) c_int,
    create_tensor: *const fn (*?*anyopaque) callconv(.c) c_int,
    destroy_tensor: *const fn (?*anyopaque) callconv(.c) c_int,
    set_tensor_ex: *const fn (?*anyopaque, c_int, c_int, c_int, [*]const c_int) callconv(.c) c_int,
    create_filter: *const fn (*?*anyopaque) callconv(.c) c_int,
    destroy_filter: *const fn (?*anyopaque) callconv(.c) c_int,
    set_filter: *const fn (?*anyopaque, c_int, c_int, c_int, [*]const c_int) callconv(.c) c_int,
    create_conv: *const fn (*?*anyopaque) callconv(.c) c_int,
    destroy_conv: *const fn (?*anyopaque) callconv(.c) c_int,
    set_conv: *const fn (?*anyopaque, c_int, [*]const c_int, [*]const c_int, [*]const c_int, c_int, c_int) callconv(.c) c_int,
    set_math: *const fn (?*anyopaque, c_int) callconv(.c) c_int,
    workspace: *const fn (?*anyopaque, ?*anyopaque, ?*anyopaque, ?*anyopaque, ?*anyopaque, c_int, *usize) callconv(.c) c_int,
    conv_fwd: *const fn (?*anyopaque, *const anyopaque, ?*anyopaque, *anyopaque, ?*anyopaque, *anyopaque, ?*anyopaque, c_int, ?*anyopaque, usize, *const anyopaque, ?*anyopaque, *anyopaque) callconv(.c) c_int,
    add_tensor: *const fn (?*anyopaque, *const anyopaque, ?*anyopaque, *anyopaque, *const anyopaque, ?*anyopaque, *anyopaque) callconv(.c) c_int,
    err_str: *const fn (c_int) callconv(.c) [*:0]const u8,
    malloc: *const fn (*?*anyopaque, usize) callconv(.c) c_int,
    free: *const fn (?*anyopaque) callconv(.c) c_int,
};

var cudnn_so: ?*anyopaque = null;
var api_cache: ?Cudnn = null;
var handle_cache: ?*anyopaque = null;

fn loadSo(name: [:0]const u8) ?*anyopaque {
    return std.c.dlopen(name, .{ .LAZY = true, .GLOBAL = true });
}

fn sym(so: ?*anyopaque, name: [:0]const u8) ?*const anyopaque {
    if (std.c.dlsym(so, name)) |p| return p;
    return std.c.dlsym(null, name);
}

fn fnPtr(comptime T: type, p: *const anyopaque) T {
    return @ptrCast(@alignCast(p));
}

fn loadApi() !Cudnn {
    if (api_cache) |a| return a;
    const so = loadSo("libcudnn.so.9") orelse loadSo("libcudnn.so") orelse null;
    cudnn_so = so;
    const g = struct {
        fn req(h: ?*anyopaque, name: [:0]const u8) !*const anyopaque {
            return sym(h, name) orelse {
                log.err("missing symbol {s}", .{name});
                return error.CudnnSymbol;
            };
        }
    };
    const rt = loadSo("libcudart.so.13") orelse so;
    const a = Cudnn{
        .create = fnPtr(@TypeOf(@as(Cudnn, undefined).create), try g.req(so, "cudnnCreate")),
        .set_stream = fnPtr(@TypeOf(@as(Cudnn, undefined).set_stream), try g.req(so, "cudnnSetStream")),
        .create_tensor = fnPtr(@TypeOf(@as(Cudnn, undefined).create_tensor), try g.req(so, "cudnnCreateTensorDescriptor")),
        .destroy_tensor = fnPtr(@TypeOf(@as(Cudnn, undefined).destroy_tensor), try g.req(so, "cudnnDestroyTensorDescriptor")),
        .set_tensor_ex = fnPtr(@TypeOf(@as(Cudnn, undefined).set_tensor_ex), try g.req(so, "cudnnSetTensorNdDescriptorEx")),
        .create_filter = fnPtr(@TypeOf(@as(Cudnn, undefined).create_filter), try g.req(so, "cudnnCreateFilterDescriptor")),
        .destroy_filter = fnPtr(@TypeOf(@as(Cudnn, undefined).destroy_filter), try g.req(so, "cudnnDestroyFilterDescriptor")),
        .set_filter = fnPtr(@TypeOf(@as(Cudnn, undefined).set_filter), try g.req(so, "cudnnSetFilterNdDescriptor")),
        .create_conv = fnPtr(@TypeOf(@as(Cudnn, undefined).create_conv), try g.req(so, "cudnnCreateConvolutionDescriptor")),
        .destroy_conv = fnPtr(@TypeOf(@as(Cudnn, undefined).destroy_conv), try g.req(so, "cudnnDestroyConvolutionDescriptor")),
        .set_conv = fnPtr(@TypeOf(@as(Cudnn, undefined).set_conv), try g.req(so, "cudnnSetConvolutionNdDescriptor")),
        .set_math = fnPtr(@TypeOf(@as(Cudnn, undefined).set_math), try g.req(so, "cudnnSetConvolutionMathType")),
        .workspace = fnPtr(@TypeOf(@as(Cudnn, undefined).workspace), try g.req(so, "cudnnGetConvolutionForwardWorkspaceSize")),
        .conv_fwd = fnPtr(@TypeOf(@as(Cudnn, undefined).conv_fwd), try g.req(so, "cudnnConvolutionForward")),
        .add_tensor = fnPtr(@TypeOf(@as(Cudnn, undefined).add_tensor), try g.req(so, "cudnnAddTensor")),
        .err_str = fnPtr(@TypeOf(@as(Cudnn, undefined).err_str), try g.req(so, "cudnnGetErrorString")),
        .malloc = fnPtr(@TypeOf(@as(Cudnn, undefined).malloc), try g.req(rt, "cudaMalloc")),
        .free = fnPtr(@TypeOf(@as(Cudnn, undefined).free), try g.req(rt, "cudaFree")),
    };
    api_cache = a;
    return a;
}

fn check(api: Cudnn, st: c_int, what: []const u8) !void {
    if (st == 0) return;
    log.err("{s}: {s} ({d})", .{ what, api.err_str(st), st });
    return error.CudnnFailed;
}

fn cudnnPatchEmbed(
    stream: *anyopaque,
    x: zml.pjrtx.CustomCallBuffer,
    w: zml.pjrtx.CustomCallBuffer,
    bias: zml.pjrtx.CustomCallBuffer,
    y: zml.pjrtx.CustomCallBuffer,
) !void {
    if (x.shape.dtype() != .bf16 or w.shape.dtype() != .bf16 or y.shape.dtype() != .bf16)
        return error.VisionEmbedDtype;
    if (x.shape.rank() != 5 or w.shape.rank() != 5 or y.shape.rank() != 2)
        return error.VisionEmbedRank;
    const n: c_int = @intCast(x.shape.dim(0));
    const ci: c_int = @intCast(x.shape.dim(1));
    const kt: c_int = @intCast(x.shape.dim(2));
    const kh: c_int = @intCast(x.shape.dim(3));
    const kw: c_int = @intCast(x.shape.dim(4));
    const co: c_int = @intCast(w.shape.dim(0));
    if (w.shape.dim(1) != x.shape.dim(1) or y.shape.dim(0) != x.shape.dim(0) or y.shape.dim(1) != w.shape.dim(0))
        return error.VisionEmbedShape;

    const api = try loadApi();
    if (handle_cache == null) {
        var h: ?*anyopaque = null;
        try check(api, api.create(&h), "cudnnCreate");
        handle_cache = h;
    }
    const handle = handle_cache;
    try check(api, api.set_stream(handle, stream), "cudnnSetStream");

    const nchw: c_int = 0;
    const dt_bf16: c_int = 9;
    const dt_f32: c_int = 0;
    const xcorr: c_int = 1;
    const tensor_op: c_int = 1;
    const algo: c_int = 1;

    var x_desc: ?*anyopaque = null;
    var y_desc: ?*anyopaque = null;
    var b_desc: ?*anyopaque = null;
    var f_desc: ?*anyopaque = null;
    var c_desc: ?*anyopaque = null;
    try check(api, api.create_tensor(&x_desc), "create x");
    try check(api, api.create_tensor(&y_desc), "create y");
    try check(api, api.create_tensor(&b_desc), "create bias");
    try check(api, api.create_filter(&f_desc), "create filter");
    try check(api, api.create_conv(&c_desc), "create conv");
    defer {
        _ = api.destroy_tensor(x_desc);
        _ = api.destroy_tensor(y_desc);
        _ = api.destroy_tensor(b_desc);
        _ = api.destroy_filter(f_desc);
        _ = api.destroy_conv(c_desc);
    }

    const x_dims = [_]c_int{ n, ci, kt, kh, kw };
    const y_dims = [_]c_int{ n, co, 1, 1, 1 };
    const b_dims = [_]c_int{ 1, co, 1, 1, 1 };
    const f_dims = [_]c_int{ co, ci, kt, kh, kw };
    try check(api, api.set_tensor_ex(x_desc, nchw, dt_bf16, 5, &x_dims), "set x");
    try check(api, api.set_tensor_ex(y_desc, nchw, dt_bf16, 5, &y_dims), "set y");
    try check(api, api.set_tensor_ex(b_desc, nchw, dt_bf16, 5, &b_dims), "set bias");
    try check(api, api.set_filter(f_desc, dt_bf16, nchw, 5, &f_dims), "set filter");

    const zeros = [_]c_int{ 0, 0, 0 };
    const stride = [_]c_int{ kt, kh, kw };
    const dil = [_]c_int{ 1, 1, 1 };
    try check(api, api.set_conv(c_desc, 3, &zeros, &stride, &dil, xcorr, dt_f32), "set conv");
    try check(api, api.set_math(c_desc, tensor_op), "set math");

    var ws_bytes: usize = 0;
    try check(api, api.workspace(handle, x_desc, f_desc, c_desc, y_desc, algo, &ws_bytes), "workspace");
    var ws: ?*anyopaque = null;
    if (ws_bytes > 0) {
        const st = api.malloc(&ws, ws_bytes);
        if (st != 0) return error.CudaMalloc;
    }
    defer if (ws) |p| {
        _ = api.free(p);
    };

    const alpha: f32 = 1;
    const beta0: f32 = 0;
    const beta1: f32 = 1;
    try check(api, api.conv_fwd(
        handle,
        &alpha,
        x_desc,
        x.ptr,
        f_desc,
        w.ptr,
        c_desc,
        algo,
        ws,
        ws_bytes,
        &beta0,
        y_desc,
        y.ptr,
    ), "conv");
    try check(api, api.add_tensor(handle, &alpha, b_desc, bias.ptr, &beta1, y_desc, y.ptr), "bias");
}
