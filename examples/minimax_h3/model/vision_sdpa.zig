const std = @import("std");

const zml = @import("zml");

const log = std.log.scoped(.minimax_h3_vision);

const Input = struct {
    q: zml.Tensor,
    k: zml.Tensor,
    v: zml.Tensor,
};
const Output = struct { o: zml.Shape };
const Attrs = struct { scale: f32 };

/// Tiled online-softmax. Native head_dim=72; full-seq `zml.nn.sdpa` is S².
const SdpaCall = zml.ops.CustomCall(Input, Output, Attrs, sdpaFfi, .{
    .name = "h3$vision_sdpa",
    .sharding_aware = false,
    .has_side_effect = false,
});

pub fn register(platform: *const zml.Platform) !void {
    try SdpaCall.register(platform);
}

pub fn forward(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, scale: f32) zml.Tensor {
    const q_t = q.transpose(.{ .b, .h, .q, .hd });
    const k_t = k.transpose(.{ .b, .h, .k, .hd });
    const v_t = v.transpose(.{ .b, .h, .k, .hd });
    const o = SdpaCall.call(
        .{ .q = q_t, .k = k_t, .v = v_t },
        .{ .o = q_t.shape() },
        .{ .scale = scale },
    ).o;
    return o.transpose(q.shape());
}

fn sdpaFfi(
    call_frame: *zml.pjrt.ffi.CallFrame,
    input: zml.pjrtx.TensorToCustomCallBuffer(Input),
    output: zml.pjrtx.ShapeToCustomCallBuffer(Output),
    attrs: Attrs,
) !?*zml.pjrt.ffi.Error {
    const stream = call_frame.stream() orelse return error.NoCudaStream;
    try launch(stream, input.q, input.k, input.v, output.o, attrs.scale);
    return null;
}

const Br = 64;
const Dmax = 80;

const kernel_src =
    \\__device__ inline float bf16_to_f32(unsigned short x) {
    \\    return __uint_as_float(((unsigned int)x) << 16);
    \\}
    \\__device__ inline unsigned short f32_to_bf16(float x) {
    \\    unsigned int u = __float_as_uint(x);
    \\    unsigned int bias = ((u >> 16) & 1u) + 0x7FFFu;
    \\    return (unsigned short)((u + bias) >> 16);
    \\}
    \\extern "C" __global__ void h3_vision_sdpa(
    \\    const unsigned short* __restrict__ Q,
    \\    const unsigned short* __restrict__ K,
    \\    const unsigned short* __restrict__ V,
    \\    unsigned short* __restrict__ O,
    \\    int B, int H, int S, int D, float scale)
    \\{
    \\    const int bh = blockIdx.x;
    \\    const int qt = blockIdx.y;
    \\    const int b = bh / H;
    \\    const int h = bh % H;
    \\    const int t = threadIdx.x;
    \\    const int q0 = qt * 64 + t;
    \\    const int valid = q0 < S;
    \\    __shared__ float Ks[64 * 80];
    \\    __shared__ float Vs[64 * 80];
    \\    const int base = ((b * H + h) * S) * D;
    \\    float qrow[80];
    \\    float acc[80];
    \\    for (int d = 0; d < 80; d++) { qrow[d] = 0; acc[d] = 0; }
    \\    if (valid) {
    \\        const unsigned short* qptr = Q + base + q0 * D;
    \\        for (int d = 0; d < D; d++) qrow[d] = bf16_to_f32(qptr[d]);
    \\    }
    \\    float m = -1.0e30f;
    \\    float l = 0.f;
    \\    for (int k0 = 0; k0 < S; k0 += 64) {
    \\        const int kc = (k0 + 64 <= S) ? 64 : (S - k0);
    \\        if (t < kc) {
    \\            const unsigned short* kptr = K + base + (k0 + t) * D;
    \\            const unsigned short* vptr = V + base + (k0 + t) * D;
    \\            for (int d = 0; d < D; d++) {
    \\                Ks[t * 80 + d] = bf16_to_f32(kptr[d]);
    \\                Vs[t * 80 + d] = bf16_to_f32(vptr[d]);
    \\            }
    \\        }
    \\        __syncthreads();
    \\        if (valid) {
    \\            float srow[64];
    \\            float m_tile = -1.0e30f;
    \\            for (int c = 0; c < kc; c++) {
    \\                float dot = 0.f;
    \\                for (int d = 0; d < D; d++) dot += qrow[d] * Ks[c * 80 + d];
    \\                srow[c] = dot * scale;
    \\                m_tile = fmaxf(m_tile, srow[c]);
    \\            }
    \\            const float m_new = fmaxf(m, m_tile);
    \\            const float alpha = __expf(m - m_new);
    \\            l *= alpha;
    \\            for (int d = 0; d < D; d++) acc[d] *= alpha;
    \\            for (int c = 0; c < kc; c++) {
    \\                const float p = __expf(srow[c] - m_new);
    \\                l += p;
    \\                for (int d = 0; d < D; d++) acc[d] += p * Vs[c * 80 + d];
    \\            }
    \\            m = m_new;
    \\        }
    \\        __syncthreads();
    \\    }
    \\    if (valid) {
    \\        unsigned short* optr = O + base + q0 * D;
    \\        const float inv = 1.f / l;
    \\        for (int d = 0; d < D; d++) optr[d] = f32_to_bf16(acc[d] * inv);
    \\    }
    \\}
;

const Nvrtc = struct {
    create: *const fn (*?*anyopaque, [*:0]const u8, [*:0]const u8, c_int, ?[*]const [*:0]const u8, ?[*]const [*:0]const u8) callconv(.c) c_int,
    compile: *const fn (?*anyopaque, c_int, [*]const [*:0]const u8) callconv(.c) c_int,
    log_size: *const fn (?*anyopaque, *usize) callconv(.c) c_int,
    log: *const fn (?*anyopaque, [*]u8) callconv(.c) c_int,
    ptx_size: *const fn (?*anyopaque, *usize) callconv(.c) c_int,
    ptx: *const fn (?*anyopaque, [*]u8) callconv(.c) c_int,
    destroy: *const fn (*?*anyopaque) callconv(.c) c_int,
    err: *const fn (c_int) callconv(.c) [*:0]const u8,
};

const Driver = struct {
    init: *const fn (c_uint) callconv(.c) c_int,
    module_load: *const fn (*?*anyopaque, *const anyopaque) callconv(.c) c_int,
    module_fn: *const fn (*?*anyopaque, ?*anyopaque, [*:0]const u8) callconv(.c) c_int,
    launch: *const fn (?*anyopaque, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, ?*anyopaque, [*]?*anyopaque, ?[*]?*anyopaque) callconv(.c) c_int,
    device_attr: *const fn (*c_int, c_int, c_int) callconv(.c) c_int,
    get_dev: *const fn (*c_int) callconv(.c) c_int,
    set_dev: *const fn (c_int) callconv(.c) c_int,
};

const KernelSlot = struct { dev: c_int = -1, fn_ptr: ?*anyopaque = null };

var nvrtc_cache: ?Nvrtc = null;
var driver_cache: ?Driver = null;
var ptx_cache: ?[]u8 = null;
var kernel_slots: [4]KernelSlot = [_]KernelSlot{.{}} ** 4;

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

fn req(so: ?*anyopaque, name: [:0]const u8) !*const anyopaque {
    return sym(so, name) orelse {
        log.err("missing symbol {s}", .{name});
        return error.CudaSymbol;
    };
}

fn loadNvrtc() !Nvrtc {
    if (nvrtc_cache) |a| return a;
    const so = loadSo("libnvrtc.so.13") orelse loadSo("libnvrtc.so");
    const a = Nvrtc{
        .create = fnPtr(@TypeOf(@as(Nvrtc, undefined).create), try req(so, "nvrtcCreateProgram")),
        .compile = fnPtr(@TypeOf(@as(Nvrtc, undefined).compile), try req(so, "nvrtcCompileProgram")),
        .log_size = fnPtr(@TypeOf(@as(Nvrtc, undefined).log_size), try req(so, "nvrtcGetProgramLogSize")),
        .log = fnPtr(@TypeOf(@as(Nvrtc, undefined).log), try req(so, "nvrtcGetProgramLog")),
        .ptx_size = fnPtr(@TypeOf(@as(Nvrtc, undefined).ptx_size), try req(so, "nvrtcGetPTXSize")),
        .ptx = fnPtr(@TypeOf(@as(Nvrtc, undefined).ptx), try req(so, "nvrtcGetPTX")),
        .destroy = fnPtr(@TypeOf(@as(Nvrtc, undefined).destroy), try req(so, "nvrtcDestroyProgram")),
        .err = fnPtr(@TypeOf(@as(Nvrtc, undefined).err), try req(so, "nvrtcGetErrorString")),
    };
    nvrtc_cache = a;
    return a;
}

fn loadDriver() !Driver {
    if (driver_cache) |a| return a;
    const so = loadSo("libcuda.so.1") orelse loadSo("libcuda.so");
    const rt = loadSo("libcudart.so.13") orelse loadSo("libcudart.so");
    const a = Driver{
        .init = fnPtr(@TypeOf(@as(Driver, undefined).init), try req(so, "cuInit")),
        .module_load = fnPtr(@TypeOf(@as(Driver, undefined).module_load), try req(so, "cuModuleLoadData")),
        .module_fn = fnPtr(@TypeOf(@as(Driver, undefined).module_fn), try req(so, "cuModuleGetFunction")),
        .launch = fnPtr(@TypeOf(@as(Driver, undefined).launch), try req(so, "cuLaunchKernel")),
        .device_attr = fnPtr(@TypeOf(@as(Driver, undefined).device_attr), try req(rt, "cudaDeviceGetAttribute")),
        .get_dev = fnPtr(@TypeOf(@as(Driver, undefined).get_dev), try req(rt, "cudaGetDevice")),
        .set_dev = fnPtr(@TypeOf(@as(Driver, undefined).set_dev), try req(rt, "cudaSetDevice")),
    };
    driver_cache = a;
    return a;
}

fn checkNvrtc(api: Nvrtc, st: c_int, what: []const u8) !void {
    if (st == 0) return;
    log.err("{s}: {s} ({d})", .{ what, api.err(st), st });
    return error.NvrtcFailed;
}

fn compilePtx() ![]const u8 {
    if (ptx_cache) |p| return p;
    const nvrtc = try loadNvrtc();
    const drv = try loadDriver();
    if (drv.init(0) != 0) return error.CudaInit;

    var major: c_int = 0;
    var minor: c_int = 0;
    if (drv.device_attr(&major, 75, 0) != 0) return error.CudaAttr;
    if (drv.device_attr(&minor, 76, 0) != 0) return error.CudaAttr;
    var arch_buf: [32]u8 = undefined;
    const arch = try std.fmt.bufPrintZ(&arch_buf, "--gpu-architecture=sm_{d}{d}", .{ major, minor });

    var prog: ?*anyopaque = null;
    try checkNvrtc(nvrtc, nvrtc.create(&prog, kernel_src, "h3_vision_sdpa.cu", 0, null, null), "nvrtcCreateProgram");
    defer _ = nvrtc.destroy(&prog);

    const opts = [_][*:0]const u8{ arch.ptr, "--std=c++17" };
    const st = nvrtc.compile(prog, opts.len, &opts);
    if (st != 0) {
        var n: usize = 0;
        _ = nvrtc.log_size(prog, &n);
        if (n > 1) {
            var log_buf: [4096]u8 = undefined;
            const take = @min(n, log_buf.len);
            _ = nvrtc.log(prog, &log_buf);
            log.err("nvrtc: {s}", .{log_buf[0 .. take - 1]});
        }
        try checkNvrtc(nvrtc, st, "nvrtcCompileProgram");
    }

    var ptx_n: usize = 0;
    try checkNvrtc(nvrtc, nvrtc.ptx_size(prog, &ptx_n), "nvrtcGetPTXSize");
    const ptx = try std.heap.c_allocator.alloc(u8, ptx_n);
    try checkNvrtc(nvrtc, nvrtc.ptx(prog, ptx.ptr), "nvrtcGetPTX");
    ptx_cache = ptx;
    return ptx;
}

fn kernelForDev(drv: Driver, dev: c_int) !?*anyopaque {
    for (kernel_slots) |s| {
        if (s.dev == dev and s.fn_ptr != null) return s.fn_ptr;
    }
    if (drv.set_dev(dev) != 0) return error.CudaSetDevice;
    const ptx = try compilePtx();
    var mod: ?*anyopaque = null;
    if (drv.module_load(&mod, ptx.ptr) != 0) return error.CuModuleLoad;
    var fn_ptr: ?*anyopaque = null;
    if (drv.module_fn(&fn_ptr, mod, "h3_vision_sdpa") != 0) return error.CuModuleFn;
    for (&kernel_slots) |*s| {
        if (s.fn_ptr == null) {
            s.* = .{ .dev = dev, .fn_ptr = fn_ptr };
            return fn_ptr;
        }
    }
    return error.VisionSdpaDevSlots;
}

fn launch(
    stream: *const anyopaque,
    q: zml.pjrtx.CustomCallBuffer,
    k: zml.pjrtx.CustomCallBuffer,
    v: zml.pjrtx.CustomCallBuffer,
    o: zml.pjrtx.CustomCallBuffer,
    scale: f32,
) !void {
    if (q.shape.dtype() != .bf16 or k.shape.dtype() != .bf16 or v.shape.dtype() != .bf16 or o.shape.dtype() != .bf16)
        return error.VisionSdpaDtype;
    if (q.shape.rank() != 4) return error.VisionSdpaRank;
    const b: i32 = @intCast(q.shape.dim(0));
    const h: i32 = @intCast(q.shape.dim(1));
    const s: i32 = @intCast(q.shape.dim(2));
    const d: i32 = @intCast(q.shape.dim(3));
    if (d > Dmax or d <= 0) return error.VisionSdpaHeadDim;
    if (k.shape.dim(2) != s or v.shape.dim(2) != s) return error.VisionSdpaShape;

    const drv = try loadDriver();
    if (drv.init(0) != 0) return error.CudaInit;
    var dev: c_int = 0;
    if (drv.get_dev(&dev) != 0) return error.CudaGetDevice;
    const func = (try kernelForDev(drv, dev)) orelse return error.VisionSdpaKernel;
    var q_ptr = q.ptr;
    var k_ptr = k.ptr;
    var v_ptr = v.ptr;
    var o_ptr = o.ptr;
    var b_arg = b;
    var h_arg = h;
    var s_arg = s;
    var d_arg = d;
    var scale_arg = scale;
    var args = [_]?*anyopaque{
        @ptrCast(&q_ptr),
        @ptrCast(&k_ptr),
        @ptrCast(&v_ptr),
        @ptrCast(&o_ptr),
        @ptrCast(&b_arg),
        @ptrCast(&h_arg),
        @ptrCast(&s_arg),
        @ptrCast(&d_arg),
        @ptrCast(&scale_arg),
    };
    const gx: c_uint = @intCast(b * h);
    const gy: c_uint = @intCast(@divFloor(s + Br - 1, Br));
    const st = drv.launch(func, gx, gy, 1, Br, 1, 1, 0, @constCast(stream), &args, null);
    if (st != 0) {
        log.err("cuLaunchKernel: {d}", .{st});
        return error.CuLaunch;
    }
}
