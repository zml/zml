const std = @import("std");

const c = @import("c");
const stdx = @import("stdx");
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
        _ = transa; // autofix
        const transb = [_]cublas_gg.cublasOperation_t{cublas_gg.CUBLAS_OP_N};
        _ = transb; // autofix
        const m = [_]c_int{@intCast(A.dim(0))}; // nbre tok
        _ = m; // autofix
        const k = [_]c_int{@intCast(A.dim(1))}; // dim
        _ = k; // autofix
        const n = [_]c_int{@intCast(B.dim(1))}; // dim out
        _ = n; // autofix
        const group_size = [_]c_int{ 1, 1, 1 };
        _ = group_size; // autofix

        const out_shape = zml.Shape.init(.{ @divExact(A.dim(0), group_count), B.dim(1) }, A.dtype());

        return zml.Tensor.gemmGroupedBatched(A, B, tokens_per_exp, .{
            .alpha = 1.0,
            .beta = 1.0,
            .group_count = group_count,
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
    const num_exp = 4;

    const M: usize = 24;
    const K: usize = 12;
    const N: usize = 10;

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
        for (0..num_exp) |i| {
            a[i] = @as(f32, @floatFromInt(i)) * -1.2;
        }
    }
    {
        const b = B_host.items(f32);
        for (0..b.len) |i| {
            // Alternate signs and vary magnitude to make mistakes obvious.
            b[i] = @as(f32, @floatFromInt(i));
        }
    }

    const data_u32: [num_exp]u32 = .{ 5, 2, 0, 3 };
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

    // Column-major CPU reference: toks_per_exp[i] is the number of output columns for expert i.
    // This matches the grouped GEMM custom call semantics used by the kernel.
    const output = try groupedGemmCpuColMajor(allocator, A_host, B_host, tokens_per_exp_host, 1.0, 0.0);
    defer output.free(allocator);

    // Print CPU reference output (column-major), formatted by rows.
    for (0..M) |row| {
        try w.print("C_cpu[row={d}] ", .{row});
        for (0..N) |col| {
            const idx = col * M + row; // column-major indexing
            const v: f32 = @bitCast(output.items(f32)[idx]);
            try w.print("{d:8.4} ", .{v});
        }
        try w.print("\n", .{});
    }

    for (0..M * N) |idx| {
        stdx.debug.assert(C_host.items(f32)[idx] == output.items(f32)[idx], "missmatch at id : {d}, in kernel output : {d}, in cpu output : {d}", .{ idx, C_host.items(f32)[idx], output.items(f32)[idx] });
    }
    log.info("All good", .{});
}

fn gemmRowMajor(
    comptime T: type,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    A: []const T, // m*k row-major
    B: []const T, // k*n row-major
    beta: T,
    C: []T, // m*n row-major (in/out)
) void {
    std.debug.assert(A.len == m * k);
    std.debug.assert(B.len == k * n);
    std.debug.assert(C.len == m * n);

    for (0..m) |i| {
        for (0..n) |j| {
            var acc: T = 0;
            for (0..k) |p| {
                acc += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = alpha * acc + beta * C[i * n + j];
        }
    }
}

fn gemmColMajorNoTrans(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    A: []const f32,
    lda: usize,
    B: []const f32,
    ldb: usize,
    beta: f32,
    C: []f32,
    ldc: usize,
) void {
    std.debug.assert(lda >= m);
    std.debug.assert(ldb >= k);
    std.debug.assert(ldc >= m);

    // C(i,j) = alpha * sum_p A(i,p)*B(p,j) + beta*C(i,j)
    for (0..n) |j| {
        for (0..m) |i| {
            var acc: f32 = 0;
            for (0..k) |p| {
                const a_ip = A[p * lda + i]; // (i,p)
                const b_pj = B[j * ldb + p]; // (p,j)
                acc += a_ip * b_pj;
            }
            const c_idx = j * ldc + i;
            C[c_idx] = alpha * acc + beta * C[c_idx];
        }
    }
}

pub fn groupedGemmCpuColMajor(
    allocator: std.mem.Allocator,
    a: zml.Slice, // f32, treated as concat of per-expert (M x K) col-major blocks
    b: zml.Slice, // f32, (K x N_total) col-major
    toks_per_exp: zml.Slice, // u32[num_exp], per-expert N_i (columns)
    alpha: f32,
    beta: f32,
) !zml.Slice {
    std.debug.assert(a.shape.dtype() == .f32);
    std.debug.assert(b.shape.dtype() == .f32);
    std.debug.assert(toks_per_exp.shape.dtype() == .u32);

    const num_exp: usize = @intCast(toks_per_exp.shape.dim(0));

    const a_rows_total: usize = @intCast(a.shape.dim(0)); // expected num_exp*M
    const K: usize = @intCast(a.shape.dim(1));

    std.debug.assert(@as(usize, @intCast(b.shape.dim(0))) == K);
    const N_total: usize = @intCast(b.shape.dim(1));

    std.debug.assert(num_exp > 0);
    std.debug.assert(a_rows_total % num_exp == 0);
    const M: usize = @divExact(a_rows_total, num_exp);

    // Validate tokens sum equals N_total
    const toks_u32 = toks_per_exp.items(u32);
    var n_sum: usize = 0;
    for (0..num_exp) |i| n_sum += @as(usize, @intCast(toks_u32[i]));
    std.debug.assert(n_sum == N_total);

    // Leading dimensions (column-major)
    const lda: usize = M;
    const ldb: usize = K;
    const ldc: usize = M;

    // Grab raw buffers
    const A_all = a.items(f32);
    const B_all = b.items(f32);

    // A is interpreted as concatenation of per-expert blocks, each lda*K elements.
    std.debug.assert(A_all.len == num_exp * (lda * K));
    std.debug.assert(B_all.len >= ldb * N_total);

    // Allocate output C (M x N_total) column-major
    var c_out = try zml.Slice.alloc(allocator, zml.Shape.init(.{ M, N_total }, .f32));
    // Initialize output (important if beta != 0)
    @memset(c_out.items(f32), 0);

    const C_all = c_out.items(f32);

    var col_off: usize = 0;
    for (0..num_exp) |gi| {
        const n_g: usize = @as(usize, @intCast(toks_u32[gi]));
        if (n_g == 0) continue;

        // A block for this expert (M x K), col-major with lda=M
        const A_g = A_all[gi * (lda * K) .. (gi + 1) * (lda * K)];

        // B starts at column col_off, so pointer offset = col_off * ldb
        const B_g = B_all[col_off * ldb ..];

        // C starts at column col_off, so pointer offset = col_off * ldc
        const C_g = C_all[col_off * ldc ..];

        gemmColMajorNoTrans(
            M,
            n_g,
            K,
            alpha,
            A_g,
            lda,
            B_g,
            ldb,
            beta,
            C_g,
            ldc,
        );

        col_off += n_g;
    }

    std.debug.assert(col_off == N_total);
    return c_out;
}
