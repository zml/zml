const std = @import("std");

const c = @import("c");
const gg = @import("platforms/cuda/grouped_gemm");
const platforms = @import("platforms");
const stdx = @import("stdx");

const zml = @import("../zml.zig");
const ffi = zml.pjrt.ffi;
const Tensor = zml.Tensor;

const log = std.log.scoped(.@"zml/grouped_gemm/cublas");

pub const cublasComputeType_t = c_int;
pub const CUBLAS_COMPUTE_32F: cublasComputeType_t = 68;

fn sliceAt(comptime T: type, base: [*]u8, offset: usize, len: usize) []T {
    const p: [*]T = @ptrCast(@alignCast(base + offset));
    return p[0..len];
}

const HostWorkspaceView = struct {
    transa: []gg.cublasOperation_t,
    transb: []gg.cublasOperation_t,
    m: []c_int,
    n: []c_int,
    k: []c_int,
    lda: []c_int,
    ldb: []c_int,
    ldc: []c_int,
    grp: []c_int,
    alpha: []f32,
    beta: []f32,
    tokens_copy: []c_int,
    h_ptr_A: []?*const anyopaque,
    h_ptr_B: []?*const anyopaque,
    h_ptr_C: []?*anyopaque,

    fn init(base: [*]u8, layout: anytype, group_count: usize) HostWorkspaceView {
        return .{
            .transa = sliceAt(gg.cublasOperation_t, base, layout.offset_transa, group_count),
            .transb = sliceAt(gg.cublasOperation_t, base, layout.offset_transb, group_count),

            .m = sliceAt(c_int, base, layout.offset_m, group_count),
            .n = sliceAt(c_int, base, layout.offset_n, group_count),
            .k = sliceAt(c_int, base, layout.offset_k, group_count),

            .lda = sliceAt(c_int, base, layout.offset_lda, group_count),
            .ldb = sliceAt(c_int, base, layout.offset_ldb, group_count),
            .ldc = sliceAt(c_int, base, layout.offset_ldc, group_count),

            .grp = sliceAt(c_int, base, layout.offset_group_size, group_count),

            .alpha = sliceAt(f32, base, layout.offset_alpha, group_count),
            .beta = sliceAt(f32, base, layout.offset_beta, group_count),

            .tokens_copy = sliceAt(c_int, base, layout.offset_tokens_copy, group_count),

            .h_ptr_A = sliceAt(?*const anyopaque, base, layout.offset_ptr_A_host, group_count),
            .h_ptr_B = sliceAt(?*const anyopaque, base, layout.offset_ptr_B_host, group_count),
            .h_ptr_C = sliceAt(?*anyopaque, base, layout.offset_ptr_C_host, group_count),
        };
    }
};

pub fn load(allocator: std.mem.Allocator, io: std.Io) !void {
    if (comptime platforms.isEnabled(.cuda)) {
        try gg.load(allocator, io);
    }
}

pub fn register(platform: zml.Platform) !void {
    if (comptime platforms.isEnabled(.cuda)) {
        try GemmGroupedBatched.register(platform);
    }
}

pub const GemmGroupedBatched = struct {
    pub const custom_call_name: [:0]const u8 = "cublas_gemm_grouped_batched";

    pub fn run(call_frame: *ffi.CallFrame) callconv(.c) ?*ffi.Error {
        return runInner(call_frame) catch |err| b: {
            log.err("cublas_gemm_grouped_batched failed: {}", .{err});
            break :b ffi.Error.create(call_frame.api, .unknown, "cublas_gemm_grouped_batched failed");
        };
    }

    pub fn register(platform: zml.Platform) !void {
        try platform.pjrt_api.ffi().?.register(platform.pjrt_api, custom_call_name, "cuda", run, .{ .command_buffer_compatible = false });
    }

    pub const Metadata = struct {
        host_buffer: zml.Tensor,
        device_buffer: zml.Tensor,

        pub const InitOptions = struct {
            group_count: usize,
        };

        pub fn init(opts: InitOptions) Metadata {
            const ptr_bytes = opts.group_count * @sizeOf(?*anyopaque);
            // device: 3 arrays of pointers
            const device_bytes = 3 * ptr_bytes;

            // (transa/transb/m/n/k/lda/ldb/ldc/group_size/alpha/beta + A/B/C pointer arrays + host_tokens)
            const host_bytes = 128 * 1024;

            return .{
                .host_buffer = .init(.{host_bytes}, .i8),
                .device_buffer = .init(.{device_bytes}, .i8),
            };
        }

        pub fn initBuffer(self: Metadata, io: std.Io, platform: zml.Platform) !zml.Bufferized(Metadata) {
            return .{
                .host_buffer = try zml.Buffer.uninitialized(io, platform, self.host_buffer.shape(), .{ .memory = .host_pinned }),
                .device_buffer = try zml.Buffer.uninitialized(io, platform, self.device_buffer.shape(), .{ .memory = .device }),
            };
        }

        pub fn deinitBuffer(self: *zml.Bufferized(Metadata)) void {
            self.host_buffer.deinit();
            self.device_buffer.deinit();
        }
    };

    const MetadataLayout = struct {
        offset_transa: usize,
        offset_transb: usize,
        offset_m: usize,
        offset_n: usize,
        offset_k: usize,
        offset_lda: usize,
        offset_ldb: usize,
        offset_ldc: usize,
        offset_group_size: usize,
        offset_alpha: usize,
        offset_beta: usize,
        offset_tokens_copy: usize,
        // Zone temporaire pour préparer les pointeurs avant copie GPU
        offset_ptr_A_host: usize,
        offset_ptr_B_host: usize,
        offset_ptr_C_host: usize,
        total_size: usize,

        fn compute(group_count: usize) MetadataLayout {
            var offset: usize = 0;
            const int_size = @sizeOf(c_int) * group_count;
            const op_size = @sizeOf(gg.cublasOperation_t) * group_count;
            const float_size = @sizeOf(f32) * group_count;
            const ptr_size = @sizeOf(usize) * group_count; // Stockés comme entiers/pointeurs

            // Helper pour aligner les offsets (important pour SIMD/GPU access)
            const align_bytes = 8;
            const next = struct {
                fn call(off: *usize, size: usize) usize {
                    const start = std.mem.alignForward(usize, off.*, align_bytes);
                    off.* = start + size;
                    return start;
                }
            }.call;

            return .{
                .offset_transa = next(&offset, op_size),
                .offset_transb = next(&offset, op_size),
                .offset_m = next(&offset, int_size),
                .offset_n = next(&offset, int_size),
                .offset_k = next(&offset, int_size),
                .offset_lda = next(&offset, int_size),
                .offset_ldb = next(&offset, int_size),
                .offset_ldc = next(&offset, int_size),
                .offset_group_size = next(&offset, int_size),
                .offset_alpha = next(&offset, float_size),
                .offset_beta = next(&offset, float_size),
                .offset_tokens_copy = next(&offset, int_size),

                .offset_ptr_A_host = next(&offset, ptr_size),
                .offset_ptr_B_host = next(&offset, ptr_size),
                .offset_ptr_C_host = next(&offset, ptr_size),

                .total_size = offset,
            };
        }
    };

    fn runInner(call_frame: *ffi.CallFrame) !?*ffi.Error {
        if (call_frame.registeringHook()) return null;

        if (gg.cublas_gemm_grouped_batched_ex == null) {
            return error.CublasNotLoaded;
        }

        if (gg.cublas_handle == null) {
            return error.CublasHandleNotInitialized;
        }
        if (gg.cublasSetStream == null) {
            return error.CublasSetSreamNotInitialized;
        }

        const buffers = call_frame.args.buffers();
        stdx.debug.assert(buffers.len >= 5, "Expected 5 input buffers", .{});
        const A_buffer = bufferFromFfiBuffer(buffers[0]);
        const B_buffer = bufferFromFfiBuffer(buffers[1]);
        const tokens_per_exp_buffer = bufferFromFfiBuffer(buffers[2]);
        const host_buffer = bufferFromFfiBuffer(buffers[3]);
        const device_buffer = bufferFromFfiBuffer(buffers[4]);

        // log.info("ffi host_buffer.ptr={any}", .{host_buffer.ptr});

        const C_buffer = bufferFromFfiBuffer(call_frame.results.buffers()[0]);

        const C_buffer_usize = std.mem.bytesAsSlice(usize, C_buffer.buffer.slice());
        C_buffer_usize[20..];

        const group_count_cint: c_int = @intCast(getScalarAttributeAs(c_int, call_frame, "group_count") orelse return error.MissingAttribute);
        const group_count: usize = @intCast(group_count_cint);

        const alpha: f32 = getScalarAttributeAs(f32, call_frame, "alpha") orelse 1.0;
        const beta: f32 = getScalarAttributeAs(f32, call_frame, "beta") orelse 0.0;
        const computeType: gg.cublasComputeType_t = @intCast(getScalarAttributeAs(i32, call_frame, "computeType") orelse 0);

        const ctx: *ffi.ExecutionContext = @constCast(call_frame.ctx);
        const cu_stream: c.CUstream = @ptrCast(call_frame.api.stream(ctx));
        _ = gg.cublasSetStream.?(gg.cublas_handle.?, cu_stream);

        try cuda.check(cuda.streamSynchronize(cu_stream));

        const c_dev_ptr: c.CUdeviceptr = @intFromPtr(C_buffer.ptr);
        const c_size_bytes = C_buffer.shape.byteSize();
        try cuda.check(cuda.memsetAsync(c_dev_ptr, 0, c_size_bytes, cu_stream));

        try cuda.check(cuda.streamSynchronize(cu_stream));

        const host_base_u8: [*]u8 = @ptrCast(@alignCast(host_buffer.ptr.?));
        //compute offsets to know where to write in the buffer
        const layout = MetadataLayout.compute(group_count);

        if (host_buffer.shape.byteSize() < layout.total_size) {
            return error.HostWorkspaceTooSmall;
        }

        // const transa_ptr: [*]gg.cublasOperation_t = @ptrCast(@alignCast(host_base_u8 + layout.offset_transa));
        // const transb_ptr: [*]gg.cublasOperation_t = @ptrCast(@alignCast(host_base_u8 + layout.offset_transb));
        // const m_ptr: [*]c_int = @ptrCast(@alignCast(host_base_u8 + layout.offset_m));
        // const n_ptr: [*]c_int = @ptrCast(@alignCast(host_base_u8 + layout.offset_n));
        // const k_ptr: [*]c_int = @ptrCast(@alignCast(host_base_u8 + layout.offset_k));
        // const lda_ptr: [*]c_int = @ptrCast(@alignCast(host_base_u8 + layout.offset_lda));
        // const ldb_ptr: [*]c_int = @ptrCast(@alignCast(host_base_u8 + layout.offset_ldb));
        // const ldc_ptr: [*]c_int = @ptrCast(@alignCast(host_base_u8 + layout.offset_ldc));
        // const grp_ptr: [*]c_int = @ptrCast(@alignCast(host_base_u8 + layout.offset_group_size));
        // const alpha_ptr: [*]f32 = @ptrCast(@alignCast(host_base_u8 + layout.offset_alpha));
        // const beta_ptr: [*]f32 = @ptrCast(@alignCast(host_base_u8 + layout.offset_beta));

        // const tokens_copy_ptr: [*]c_int = @ptrCast(@alignCast(host_base_u8 + layout.offset_tokens_copy));

        // const h_ptr_A: [*]?*const anyopaque = @ptrCast(@alignCast(host_base_u8 + layout.offset_ptr_A_host));
        // const h_ptr_B: [*]?*const anyopaque = @ptrCast(@alignCast(host_base_u8 + layout.offset_ptr_B_host));
        // const h_ptr_C: [*]?*anyopaque = @ptrCast(@alignCast(host_base_u8 + layout.offset_ptr_C_host));

        // const trans_a = host_buffer.ptr[host_base_u8..][0..layout.offset_transa];
        // _ = trans_a; // autofix

        var ws = HostWorkspaceView.init(host_base_u8, layout, group_count);

        const src_tokens_dev: usize = @intFromPtr(tokens_per_exp_buffer.ptr.?);

        // Verif Type
        const dtype = tokens_per_exp_buffer.shape.dtype();
        if (dtype != .i32) {
            log.err("CRITICAL ERROR: tokens_per_exp expected .i32, got {any}. Cast needed upstream!", .{dtype});
            return error.UnsupportedDataType;
        }

        // Taille
        const num_elements = tokens_per_exp_buffer.shape.dim(0);
        if (num_elements < group_count) {
            log.err("CRITICAL ERROR: tokens_per_exp has {d} elements, but group_count is {d}. Read out of bounds!", .{ num_elements, group_count });
            return error.DimensionMismatch;
        }

        const size_to_copy = group_count * @sizeOf(c_int);

        // log.info("device_buffer.ptr=0x{x} bytes={d}", .{
        //     @intFromPtr(device_buffer.ptr.?),
        //     device_buffer.shape.byteSize(),
        // });

        // log.info("host_buffer.ptr=0x{x} bytes={d}", .{
        //     @intFromPtr(host_buffer.ptr.?),
        //     host_buffer.shape.byteSize(),
        // });
        // log.info("tokens_copy_ptr=0x{x} copy_bytes={d}", .{
        //     @intFromPtr(ws.tokens_copy.ptr),
        //     size_to_copy,
        // });
        // log.info("tokens_per_exp dev ptr=0x{x}", .{@intFromPtr(tokens_per_exp_buffer.ptr.?)});

        @memset(host_base_u8[0..layout.total_size], 0);
        // const allocator = std.heap.c_allocator;
        // var threaded: std.Io.Threaded = .init(allocator, .{});
        // defer threaded.deinit();

        // const io = threaded.io();
        // try io.sleep(.{ .nanoseconds = 100 }, .real);

        try cuda.check(cuda.memcpyToHostAsync(ws.tokens_copy.ptr, src_tokens_dev, size_to_copy, cu_stream));
        const Ctx = struct {
            a: usize,
            b: usize,
        };

        var ctx: Ctx = .{
            .a = 23,
            .b = 18,
        };
        try cuda.check(cuda.launchHostFunc(cu_stream, struct {
            fn call(userdata: ?*anyopaque) callconv(.c) void {
                const ctx_: *Ctx = @ptrCast(@alignCast(userdata.?));
            }
        }.call, host_base_u8));

        try cuda.check(cuda.streamSynchronize(cu_stream));
        try cuda.check(cuda.ctxSynchronize());

        // try io.sleep(.{ .nanoseconds = 100 }, .real);

        for (0..group_count) |i| {
            ws.transa[i] = gg.CUBLAS_OP_T;
            ws.transb[i] = gg.CUBLAS_OP_N;
            ws.grp[i] = 1;
            ws.alpha[i] = alpha;
            ws.beta[i] = beta;
        }

        const Ad = A_buffer.shape.dims();
        const out_dim: c_int = @intCast(Ad[0]);
        const k: c_int = @intCast(Ad[1]);
        const m = @divExact(out_dim, group_count_cint);
        const elem_size = A_buffer.shape.dtype().sizeOf();

        // Adresses GPU
        const A_base: [*]const u8 = @ptrCast(@alignCast(A_buffer.ptr));
        const B_base: [*]const u8 = @ptrCast(@alignCast(B_buffer.ptr));
        const C_base: [*]u8 = @ptrCast(@alignCast(C_buffer.ptr));

        var a_off: usize = 0;
        var b_off: usize = 0;
        var c_off: usize = 0;
        // log.info(">>>>>>>>>>>>>>>>>>>>>>>>>>", .{});
        for (0..@intCast(group_count)) |i| {
            // std.log.info("group {d}: {d}", .{ i, ws.tokens_copy[i] });
            const n = ws.tokens_copy[i];

            ws.m[i] = m;
            ws.n[i] = n;
            ws.k[i] = k;
            ws.lda[i] = k;
            ws.ldb[i] = k;
            ws.ldc[i] = m;

            ws.h_ptr_A[i] = @ptrCast(A_base + a_off);
            ws.h_ptr_B[i] = @ptrCast(B_base + b_off);
            ws.h_ptr_C[i] = @ptrCast(C_base + c_off);

            a_off += @as(usize, @intCast(ws.lda[i])) * @as(usize, @intCast(m)) * elem_size;
            b_off += @as(usize, @intCast(ws.ldb[i])) * @as(usize, @intCast(n)) * elem_size; //ng uninitialized if not synchro
            c_off += @as(usize, @intCast(ws.ldc[i])) * @as(usize, @intCast(n)) * elem_size;
            // log.info("G[{d}] A_ptr=0x{x} B_ptr=0x{x} C_ptr=0x{x}", .{
            //     i,
            //     @intFromPtr(ws.h_ptr_A[i].?),
            //     @intFromPtr(ws.h_ptr_B[i].?),
            //     @intFromPtr(ws.h_ptr_C[i].?),
            // });

            // log.info("G[{d}] m={d} n={d} k={d} lda={d} ldb={d} ldc={d}", .{
            //     i, ws.m[i], ws.n[i], ws.k[i], ws.lda[i], ws.ldb[i], ws.ldc[i],
            // });
        }

        const ptr_array_bytes = group_count * @sizeOf(?*anyopaque);
        const device_base_ptr = @intFromPtr(device_buffer.ptr.?);

        const d_A_array_ptr = device_base_ptr;
        const d_B_array_ptr = device_base_ptr + ptr_array_bytes;
        const d_C_array_ptr = device_base_ptr + (ptr_array_bytes * 2);

        if (device_buffer.shape.byteSize() < ptr_array_bytes * 3) {
            return error.DeviceWorkspaceTooSmall;
        }

        // Copie asynchrone : Host Pinned (h_ptr_X) -> Device (d_X_array_ptr)
        // Comme h_ptr_X est dans le buffer host pinned, le DMA est efficace.
        try cuda.check(cuda.memcpyToDeviceAsync(d_A_array_ptr, @ptrCast(ws.h_ptr_A.ptr), ptr_array_bytes, cu_stream));
        try cuda.check(cuda.memcpyToDeviceAsync(d_B_array_ptr, @ptrCast(ws.h_ptr_B.ptr), ptr_array_bytes, cu_stream));
        try cuda.check(cuda.memcpyToDeviceAsync(d_C_array_ptr, @ptrCast(ws.h_ptr_C.ptr), ptr_array_bytes, cu_stream));
        try cuda.check(cuda.streamSynchronize(cu_stream));

        // -----------------------------------------------------------------------
        // 6. Exécution
        // -----------------------------------------------------------------------
        const Atype: gg.cudaDataType_t = switch (A_buffer.shape.dtype()) {
            .f32 => gg.CUDA_R_32F,
            .f16 => gg.CUDA_R_16F,
            .bf16 => gg.CUDA_R_16BF,
            .f64 => gg.CUDA_R_64F,
            else => return error.UnsupportedDataType,
        };

        // Note: m_ptr, n_ptr, etc. sont passés directement.
        // cuBLAS supporte la lecture depuis Host Pinned memory (c'est le cas ici).
        // Par contre, les tableaux de pointeurs (8ème argument, etc) doivent souvent être sur Device,
        // c'est pourquoi nous passons d_A_array_ptr.
        const status = gg.cublas_gemm_grouped_batched_ex.?(
            gg.cublas_handle.?,
            ws.transa.ptr,
            ws.transb.ptr,
            ws.m.ptr,
            ws.n.ptr,
            ws.k.ptr,
            ws.alpha.ptr,
            @ptrFromInt(d_A_array_ptr),
            Atype,
            ws.lda.ptr,
            @ptrFromInt(d_B_array_ptr),
            Atype,
            ws.ldb.ptr,
            ws.beta.ptr,
            @ptrFromInt(d_C_array_ptr),
            Atype,
            ws.ldc.ptr,
            group_count_cint,
            ws.grp.ptr,
            computeType,
        );

        try cuda.check(cuda.streamSynchronize(cu_stream));
        try cuda.check(cuda.ctxSynchronize());

        if (status != gg.CUBLAS_STATUS_SUCCESS) return error.CublasError;

        return null;
    }
};

//         const td = tokens_per_exp_buffer.shape.dims();
//         stdx.debug.assert(td.len == 1, "tokens_per_exp expected rank-1, got rank={d}", .{td.len});
//         const count: usize = @intCast(td[0]);

//         // const host_tokens = arena.allocator().alloc(c_int, @intCast(group_count)) catch unreachable;

//         stdx.debug.assert(tokens_per_exp_buffer.ptr != null, "tokens_per_exp device ptr is null", .{});

//         const CUdeviceptr = usize;
//         const src_dev: CUdeviceptr = @intFromPtr(tokens_per_exp_buffer.ptr.?);

//         const bytes: usize = count * @sizeOf(c_int);

//         // try gg.cuda.check(gg.cuda.memcpyToHostAsync(host_tokens.ptr, src_dev, bytes, cu_stream));

//         const Ad = A_buffer.shape.dims();
//         const Bd = B_buffer.shape.dims();

//         stdx.debug.assert(Ad.len == 2, "Expected A to be rank-2 [m,k], got rank={d}", .{Ad.len});
//         stdx.debug.assert(Bd.len == 2, "Expected B to be rank-2 [k,n], got rank={d}", .{Bd.len});

//         const elem_size: usize = switch (A_buffer.shape.dtype()) {
//             .f16, .bf16, .f32, .f64 => |v| v.sizeOf(),
//             else => return error.UnsupportedDataType,
//         };

//         const Aarray = arena.allocator().alloc(?*const anyopaque, @intCast(group_count)) catch unreachable;
//         const Barray = arena.allocator().alloc(?*const anyopaque, @intCast(group_count)) catch unreachable;
//         const Carray = arena.allocator().alloc(?*anyopaque, @intCast(group_count)) catch unreachable;

//         @memset(Carray, null);

//         // Base pointers as bytes for byte-offset arithmetic
//         const A_base: [*]const u8 = @ptrCast(@alignCast(A_buffer.ptr));
//         const B_base: [*]const u8 = @ptrCast(@alignCast(B_buffer.ptr));
//         const C_base: [*]u8 = @ptrCast(@alignCast(C_buffer.ptr));

//         var a_off_bytes: usize = 0;
//         var b_off_bytes: usize = 0;
//         var c_off_bytes: usize = 0;

//         const m: c_int = @intCast(Ad[0]);
//         const k: c_int = @intCast(Ad[1]);

//         // We hard-code: 1 GEMM per group
//         for (0..@intCast(group_count)) |gi| {
//             transa_array[gi] = gg.CUBLAS_OP_T;
//             transb_array[gi] = gg.CUBLAS_OP_N;

//             n_array[gi] = host_tokens[gi];
//             m_array[gi] = @divExact(m, group_count);
//             k_array[gi] = k;

//             lda_array[gi] = k;
//             ldb_array[gi] = k;
//             ldc_array[gi] = @divExact(m, group_count);

//             group_size[gi] = 1;

//             const lda: usize = @intCast(lda_array[gi]);
//             const ldb: usize = @intCast(ldb_array[gi]);
//             const ldc: usize = @intCast(ldc_array[gi]);

//             const n_g: usize = @intCast(n_array[gi]);
//             const m_g: usize = @intCast(m_array[gi]);

//             const A_elems: usize = lda * m_g;
//             const B_elems: usize = ldb * n_g;
//             const C_elems: usize = ldc * n_g;

//             Aarray[gi] = @ptrCast(A_base + a_off_bytes);
//             Barray[gi] = @ptrCast(B_base + b_off_bytes);
//             Carray[gi] = @ptrCast(C_base + c_off_bytes);

//             a_off_bytes += A_elems * elem_size;
//             b_off_bytes += B_elems * elem_size;
//             c_off_bytes += C_elems * elem_size;
//         }

//         // You'll need to populate m_array, n_array, k_array, etc. from your actual data
//         const ptr_bytes: usize = @as(usize, @intCast(group_count)) * @sizeOf(?*anyopaque);

//         var d_A_array: CUdeviceptr = 0;
//         var d_B_array: CUdeviceptr = 0;
//         var d_C_array: CUdeviceptr = 0;

//         //faire a l'init
//         try gg.cuda.check(gg.cuda.memAlloc(&d_A_array, ptr_bytes));
//         defer _ = gg.cuda.memFree(d_A_array);

//         try gg.cuda.check(gg.cuda.memAlloc(&d_B_array, ptr_bytes));
//         defer _ = gg.cuda.memFree(d_B_array);

//         try gg.cuda.check(gg.cuda.memAlloc(&d_C_array, ptr_bytes));
//         defer _ = gg.cuda.memFree(d_C_array);

//         const A_src: ?*const anyopaque = @ptrCast(Aarray.ptr);
//         const B_src: ?*const anyopaque = @ptrCast(Barray.ptr);
//         const C_src: ?*const anyopaque = @ptrCast(Carray.ptr);

//         // Copy host arrays of device pointers -> device arrays
//         try gg.cuda.check(gg.cuda.memcpyToDeviceAsync(d_A_array, A_src, ptr_bytes, cu_stream));
//         try gg.cuda.check(gg.cuda.memcpyToDeviceAsync(d_B_array, B_src, ptr_bytes, cu_stream));
//         try gg.cuda.check(gg.cuda.memcpyToDeviceAsync(d_C_array, C_src, ptr_bytes, cu_stream));

//         const Atype: gg.cudaDataType_t = switch (A_buffer.shape.dtype()) {
//             .f32 => gg.CUDA_R_32F, // CUDA_R_32F
//             .f16 => gg.CUDA_R_16F, // CUDA_R_16F
//             .bf16 => gg.CUDA_R_16BF, // CUDA_R_16BF
//             .f64 => gg.CUDA_R_64F, // CUDA_R_64F
//             else => return error.UnsupportedDataType,
//         };
//         const Btype: gg.cudaDataType_t = Atype;
//         const Ctype: gg.cudaDataType_t = Atype;

//         // Call cuBLAS function
//         const status = gg.cublas_gemm_grouped_batched_ex.?(
//             gg.cublas_handle.?,
//             transa_array.ptr,
//             transb_array.ptr,
//             m_array.ptr,
//             n_array.ptr,
//             k_array.ptr,
//             alpha_array.ptr,
//             @ptrFromInt(@as(usize, @intCast(d_A_array))),
//             Atype,
//             lda_array.ptr,
//             @ptrFromInt(@as(usize, @intCast(d_B_array))),
//             Btype,
//             ldb_array.ptr,
//             beta_array.ptr,
//             @ptrFromInt(@as(usize, @intCast(d_C_array))),
//             Ctype,
//             ldc_array.ptr,
//             group_count,
//             group_size.ptr,
//             computeType,
//         );

//         if (status != gg.CUBLAS_STATUS_SUCCESS) {
//             return error.CublasError;
//         }

//         return null;
//     }
// };

fn dataTypeFromFfiDataType(ffi_dt: ffi.DataType) zml.DataType {
    return switch (ffi_dt) {
        .pred => .bool,
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

// pub fn Wrapper(comptime T: type, run_func: std.meta.DeclEnum(T)) type {
//     return struct {
//         pub fn register(platform: zml.Platform) !void {
//             try platform.pjrt_api.ffi().?.register(platform.pjrt_api, T.custom_call_name, "cuda", T.run, .{ .command_buffer_compatible = true });
//         }

//         pub fn run(call_frame: *ffi.CallFrame) callconv(.c) ?*ffi.Error {
//             return @field(T, @tagName(run_func))(call_frame) catch b: {
//                 break :b ffi.Error.create(call_frame.api.?, .unknown, "Unknown");
//             };
//         }
//     };
// }

fn shapeFromFfiBuffer(buffer: *const ffi.Buffer) zml.Shape {
    return .init(buffer.dims(), dataTypeFromFfiDataType(buffer.dtype));
}

const FfiBuffer = struct {
    buffer: ffi.Buffer,
    shape: zml.Shape,
};

fn bufferFromFfiBuffer(ffi_buffer: *const ffi.Buffer) FfiBuffer {
    const dtype = switch (ffi_buffer.dtype) {
        .f32 => zml.DataType.f32,
        .f16 => zml.DataType.f16,
        .bf16 => zml.DataType.bf16,
        .f64 => zml.DataType.f64,
        .u32 => zml.DataType.u32,
        .i32 => zml.DataType.i32,
        .i8 => zml.DataType.i8,
        else => unreachable,
    };
    return .{
        .buffer = ffi_buffer,
        .shape = zml.Shape.init(ffi_buffer.dims(), dtype),
    };
}

fn getScalarAttributeAs(comptime T: type, call_frame: *ffi.CallFrame, attribute_name: []const u8) ?T {
    const attribute = call_frame.attrs.getByName(.scalar, attribute_name) orelse return null;
    return attribute.get(T);
}

pub fn gemmGroupedBatched(
    A: Tensor,
    B: Tensor,
    tokens_per_exp: Tensor,
    host_buffer: Tensor,
    device_buffer: Tensor,
    opts: struct {
        // transa_array: []const cublasOperation_t,
        // transb_array: []const cublasOperation_t,
        // m_array: []const c_int,
        // n_array: []const c_int,
        // k_array: []const c_int,
        alpha: f32 = 1.0,
        beta: f32 = 0.0,
        group_count: c_int,
        // group_size: []const c_int,
        computeType: gg.cublasComputeType_t,
        output_shape: zml.Shape,
    },
) [3]Tensor {
    const inputs = .{ A, B, tokens_per_exp, host_buffer, device_buffer };
    return zml.ops.customCall(
        GemmGroupedBatched.custom_call_name,
        inputs,
        .{
            host_buffer.shape(),
            device_buffer.shape(),
            opts.output_shape,
        },
        .{
            .group_count = opts.group_count,
            .alpha = opts.alpha,
            .beta = opts.beta,
            .computeType = opts.computeType,
            // Note: Arrays need to be passed differently - you might need to
            // create buffers for them or use a different approach
        },
        .{ .has_side_effect = false, .output_operand_aliases = &[_]i64{ 3, 4 } },
    );
}

pub const cuda = struct {
    pub const Error = error{
        CudaError,
    };

    const memcpyToDeviceAsync = @extern(*const @TypeOf(c.cuMemcpyHtoDAsync_v2), .{ .name = "cuMemcpyHtoDAsync_v2", .linkage = .weak }).?;
    const memcpyToDevice = @extern(*const @TypeOf(c.cuMemcpyHtoD_v2), .{ .name = "cuMemcpyHtoD_v2", .linkage = .weak }).?;

    pub fn memcpyToHostAsync(dstHost: []u8, srcDevice: []const u8, stream: c.CUstream) CUresult {
        std.debug.assert(dstHost.len == srcDevice.len);
        const f = @extern(*const @TypeOf(c.cuMemcpyDtoHAsync_v2), .{ .name = "cuMemcpyDtoHAsync_v2", .linkage = .weak }).?;
        return f.?(@ptrCast(dstHost), @ptrCast(srcDevice), stream);
    }

    const memcpyToHost = @extern(*const @TypeOf(c.cuMemcpyDtoH_v2), .{ .name = "cuMemcpyDtoH_v2", .linkage = .weak }).?;

    const launchHostFunc = @extern(*const @TypeOf(c.cuLaunchHostFunc), .{ .name = "cuLaunchHostFunc", .linkage = .weak }).?;
    pub const streamSynchronize = @extern(*const @TypeOf(c.cuStreamSynchronize), .{ .name = "cuStreamSynchronize", .linkage = .weak }).?;
    pub const ctxSynchronize = @extern(*const @TypeOf(c.cuCtxSynchronize), .{ .name = "cuCtxSynchronize", .linkage = .weak }).?;

    pub const memsetAsync = @extern(*const @TypeOf(c.cuMemsetD8Async), .{ .name = "cuMemsetD8Async", .linkage = .weak }).?;

    const memAlloc = @extern(*const @TypeOf(c.cuMemAlloc_v2), .{ .name = "cuMemAlloc_v2", .linkage = .weak }).?;
    const memFree = @extern(*const @TypeOf(c.cuMemFree_v2), .{ .name = "cuMemFree_v2", .linkage = .weak }).?;
    const CUresult = c.CUresult;

    pub fn check(result: CUresult) Error!void {
        if (result == c.CUDA_SUCCESS) return;
        std.log.err("cuda error: {}", .{result});
        return Error.Failed;
    }
};
