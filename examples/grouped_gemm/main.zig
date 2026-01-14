const std = @import("std");

const c = @import("c");
const zml = @import("zml");
const ops = zml.ops;
const cublas_gg = zml.cublas_grouped_gemm;
const CublasGemmGroupedBatchedExFunc = cublas_gg.CublasGemmGroupedBatchedExFunc;
const GemmGroupedBatched = cublas_gg.GemmGroupedBatched;

const Demo = struct {
    pub fn forward(A: zml.Tensor, B: zml.Tensor, tokens_per_exp: zml.Tensor) zml.Tensor {
        // group_count=1 demo
        const group_count: c_int = 3;
        const transa = [_]cublas_gg.cublasOperation_t{cublas_gg.CUBLAS_OP_N};
        const transb = [_]cublas_gg.cublasOperation_t{cublas_gg.CUBLAS_OP_N};
        const m = [_]c_int{@intCast(A.dim(0))}; // nbre tok
        const k = [_]c_int{@intCast(A.dim(1))}; // dim
        const n = [_]c_int{@intCast(B.dim(1))}; // dim out
        const group_size = [_]c_int{ 1, 1, 1 };

        const out_shape = zml.Shape.init(.{ 256, B.dim(1) }, A.dtype());

        return zml.Tensor.gemmGroupedBatched(A, B, tokens_per_exp, .{
            .transa_array = &transa,
            .transb_array = &transb,
            .m_array = &m,
            .n_array = &n,
            .k_array = &k,
            .alpha = 1.0,
            .beta = 1.0,
            .group_count = group_count,
            .group_size = &group_size,
            .computeType = cublas_gg.CUBLAS_COMPUTE_32F,
            .output_shape = out_shape,
        });
    }
};

const log = std.log.scoped(.grouped_gemm);
pub fn main() !void {
    const allocator = std.heap.c_allocator;

    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();

    const io = threaded.io();

    zml.init();
    defer zml.deinit();

    // Auto-select platform
    const platform: zml.Platform = try .auto(io, .{});

    // Skip if not CUDA platform
    if (platform.target != .cuda) {
        log.warn("Platform is not CUDA, skipping execution. This example requires CUDA.", .{});
        return;
    }

    try GemmGroupedBatched.register(platform);
    try cublas_gg.load(allocator);

    // Shapes: A[M,K], B[K,N]
    const num_exp = 8;

    const M: usize = 4096;
    const K: usize = 1024;
    const N: usize = 128;

    const A_t: zml.Tensor = .init(.{ num_exp * M, K }, .f32);
    const B_t: zml.Tensor = .init(.{ K, N }, .f32);
    const tokens_per_exp_t: zml.Tensor = .init(.{num_exp}, .u32);

    log.info("Compiling custom call demo...", .{});
    const demo: Demo = .{};
    _ = demo; // autofix
    var exe = try platform.compileFn(allocator, io, Demo.forward, .{ A_t, B_t, tokens_per_exp_t });
    defer exe.deinit();
    log.info("Compiled", .{});

    var args = try exe.args(allocator);
    defer args.deinit(allocator);

    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    // Host data
    var A_host = try zml.Slice.alloc(allocator, zml.Shape.init(.{ num_exp * M, K }, .f32));
    defer A_host.free(allocator);
    var B_host = try zml.Slice.alloc(allocator, zml.Shape.init(.{ K, N }, .f32));
    defer B_host.free(allocator);
    var tokens_per_exp_host = try zml.Slice.alloc(allocator, zml.Shape.init(.{num_exp}, .u32));
    defer tokens_per_exp_host.free(allocator);

    // Fill with something deterministic (all ones)
    @memset(A_host.items(f32), 1.0); // 1.0 in f16
    @memset(B_host.items(f32), 1.0); // 1.0 in f16

    const data_u32: [num_exp]u32 = .{ 16, 16, 16, 16, 16, 16, 16, 16 };
    @memcpy(tokens_per_exp_host.items(u32)[0..num_exp], &data_u32);

    var A_buf: zml.Buffer = try .fromSlice(io, platform, A_host);
    defer A_buf.deinit();
    var B_buf: zml.Buffer = try .fromSlice(io, platform, B_host);
    defer B_buf.deinit();
    const tokens_per_exp_buf: zml.Buffer = try .fromSlice(io, platform, tokens_per_exp_host);
    defer tokens_per_exp_buf.deinit();
    args.set(.{ A_buf, B_buf, tokens_per_exp_buf });
    exe.callOpts(io, args, &results, .{ .wait = true });

    var C_buf: zml.Buffer = results.get(zml.Buffer);
    defer C_buf.deinit();

    var C_host = try zml.Slice.alloc(allocator, zml.Shape.init(.{ M, N }, .f32));
    defer C_host.free(allocator);
    try C_buf.toSlice(io, C_host);

    log.info("C[0,0] : {d}", .{@as(f32, @bitCast(C_host.items(f32)[100]))});
}
