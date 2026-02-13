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
        const group_count: c_int = @intCast(tokens_per_exp.dim(0));
        const transa = [_]cublas_gg.cublasOperation_t{cublas_gg.CUBLAS_OP_N};
        const transb = [_]cublas_gg.cublasOperation_t{cublas_gg.CUBLAS_OP_N};
        const m = [_]c_int{@intCast(A.dim(0))}; // nbre tok
        const k = [_]c_int{@intCast(A.dim(1))}; // dim
        const n = [_]c_int{@intCast(B.dim(1))}; // dim out
        const group_size = [_]c_int{ 1, 1, 1 };

        const out_shape = zml.Shape.init(.{ @divExact(A.dim(0), group_count), B.dim(1) }, A.dtype());

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
    const num_exp = 2;

    const M: usize = 5;
    const K: usize = 4;
    const N: usize = 3;

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
    {
        const a = A_host.items(f32);
        for (0..a.len) |i| {
            // Vary across the whole buffer with a mix of integer + fractional parts.
            a[i] = @as(f32, @floatFromInt(i));
        }
    }
    {
        const b = B_host.items(f32);
        for (0..b.len) |i| {
            // Alternate signs and vary magnitude to make mistakes obvious.
            b[i] = @floatFromInt(i + 1);
        }
    }

    const data_u32: [num_exp]u32 = .{ 1, 2 };
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
    log.info(">>>COUCOU", .{});
    _ = try C_buf.await(io);
    log.info(">>>COUCOU1", .{});
    try C_buf.toSlice(io, C_host);
    log.info(">>>COUCOU2", .{});

    // Column-major buffer print, formatted by rows:
    // buffer index = col * M + row (cuBLAS convention), but we print row-by-row for readability.
    var stdout = std.Io.File.stdout().writer(io, &.{});
    const w = &stdout.interface;

    for (0..M) |row| {
        try w.print("C[row={d}] ", .{row});
        for (0..N) |col| {
            const idx = col * M + row; // column-major indexing
            const v: f32 = @bitCast(C_host.items(f32)[idx]);
            try w.print("{d:8.4} ", .{v});
        }
        try w.print("\n", .{});
    }
}
