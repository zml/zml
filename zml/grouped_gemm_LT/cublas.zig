const std = @import("std");

const c = @import("c");
const lt = @import("platforms/cuda/grouped_gemm_LT");
const platforms = @import("platforms");
const stdx = @import("stdx");

const zml = @import("../zml.zig");
const ffi = zml.pjrt.ffi;
const Tensor = zml.Tensor;

const log = std.log.scoped(.@"zml/grouped_gemm/cublas");

pub const cublasComputeType_t = c_int;
pub const CUBLAS_COMPUTE_32F: cublasComputeType_t = 68;

var cublas_handle: ?lt.cublasHandle_t = null;

fn sliceAt(comptime T: type, base: []u8, offset: usize, len: usize) []T {
    const slice: []T = @ptrCast(@alignCast(base[offset..]));
    return slice[0..len];
}

const HostWorkspaceView = struct {
    //slices in the host

    fn init(base: []u8, layout: anytype, group_count: usize) HostWorkspaceView {
        _ = base; // autofix
        _ = layout; // autofix
        _ = group_count; // autofix
        return .{
            //build the slices
        };
    }
};

pub fn load(allocator: std.mem.Allocator, io: std.Io) !void {
    _ = io; // autofix
    if (comptime platforms.isEnabled(.cuda)) {
        try lt.load(allocator);
    }
}

pub fn register(platform: zml.Platform) !void {
    if (comptime platforms.isEnabled(.cuda)) {
        try GemmGroupedBatchedLT.register(platform);
    }
}

pub const GemmGroupedBatchedLT = struct {
    pub const custom_call_name: [:0]const u8 = "cublas_gemm_grouped_batched";

    pub fn run(call_frame: *ffi.CallFrame) callconv(.c) ?*ffi.Error {
        return runInner(call_frame) catch |err| b: {
            log.err("cublas_gemm_grouped_batched LT failed: {}", .{err});
            break :b ffi.Error.create(call_frame.api, .unknown, "cublas_gemm_grouped_batched LT failed");
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
            _ = opts; // autofix
            // const ptr_bytes = opts.group_count * @sizeOf(?*anyopaque);
            // device: 3 arrays of  p oint e rs
            const device_bytes = 24 * 1024 * 1024;

            // host datas
            const host_bytes = 10 * 1024 * 1024;

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

    const MetadataLayout = struct { //Compute offset en bytes to slice into the host buffer
        //offsets

        fn compute(group_count: usize) MetadataLayout {
            if (group_count == 0 or group_count > 1_000_000) {
                @panic("corrupted group_count");
            }
            var offset: usize = 0;
            const int_size = @sizeOf(c_int) * group_count;
            const op_size = @sizeOf(lt.cublasOperation_t) * group_count;
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
        // ... [Init, Streams, and Buffer Setup same as before] ...
        if (call_frame.registeringHook()) return null;
        if (lt.cublaslt_handle == null) return error.CublasNotInitialized;

        const ctx: *ffi.ExecutionContext = @constCast(call_frame.ctx);
        const cu_stream: c.CUstream = @ptrCast(call_frame.api.stream(ctx));

        const buffers = call_frame.args.buffers();
        const A_buffer = bufferFromFfiBuffer(buffers[0]);

        const B_buffer = bufferFromFfiBuffer(buffers[1]);
        const tokens_per_exp_buffer = bufferFromFfiBuffer(buffers[2]);
        const host_workspace = bufferFromFfiBuffer(buffers[3]);
        const device_workspace = bufferFromFfiBuffer(buffers[4]);

        const group_count: usize = @intCast(getScalarAttributeAs(c_int, call_frame, "group_count") orelse return error.MissingGroupCount);
        const alpha: f32 = getScalarAttributeAs(f32, call_frame, "alpha") orelse 1.0;
        const beta: f32 = getScalarAttributeAs(f32, call_frame, "beta") orelse 0.0;

        const C_buffer = bufferFromFfiBuffer(call_frame.results.buffers()[2]);
        // log.info("A_buffer dim 0 {}", .{A_buffer.buffer.dims()[0]});
        // log.info("A_buffer dim 1 {}", .{A_buffer.buffer.dims()[1]});
        // log.info("C_buffer dim 0 {}", .{C_buffer.buffer.dims()[0]});
        // log.info("C_buffer dim 1 {}", .{C_buffer.buffer.dims()[1]});
        // log.info("B_buffer dim 0 {}", .{B_buffer.buffer.dims()[0]});
        // log.info("B_buffer dim 1 {}", .{B_buffer.buffer.dims()[1]});
        const k_dim = A_buffer.buffer.dims()[1];
        const m_total = A_buffer.buffer.dims()[0];
        const m_dim = @divExact(m_total, @as(i64, @intCast(group_count)));

        const cuda_type: c_int = switch (A_buffer.shape.dtype()) {
            .f16 => lt.CUDA_R_16F,
            .f32 => lt.CUDA_R_32F,
            .bf16 => lt.CUDA_R_16BF,
            else => return error.UnsupportedDataType,
        };

        const tokens_slice_device = tokens_per_exp_buffer.buffer.slice();
        const tokens_bytes_len = group_count * @sizeOf(c_int);
        if (host_workspace.buffer.slice().len < tokens_bytes_len) return error.HostWorkspaceTooSmall;
        const tokens_slice_host = host_workspace.buffer.slice()[0..tokens_bytes_len];
        try cuda.check(cuda.memcpyToHostAsync(tokens_slice_host, tokens_slice_device, cu_stream));
        try cuda.check(cuda.streamSynchronize(cu_stream));
        const tokens_slice_host_cint: []c_int = @alignCast(std.mem.bytesAsSlice(c_int, tokens_slice_host));

        const device_ws_slice = device_workspace.buffer.slice();
        const ws_size = device_ws_slice.len;

        var opDesc: lt.cublasLtMatmulDesc_t = null;
        if (lt.cublasLtMatmulDescCreate.?(&opDesc, CUBLAS_COMPUTE_32F, lt.CUDA_R_32F) != lt.CUBLAS_STATUS_SUCCESS) return error.CublasDescCreateFailed;
        defer _ = lt.cublasLtMatmulDescDestroy.?(opDesc);

        const op_T: c_int = 1; // CUBLAS_OP_T
        const op_N: c_int = 0; // CUBLAS_OP_N

        const ATTR_TRANSA: c_int = 3; // CUBLASLT_MATMUL_DESC_TRANSA
        const ATTR_TRANSB: c_int = 4; // CUBLASLT_MATMUL_DESC_TRANSB

        _ = lt.cublasLtMatmulDescSetAttribute.?(opDesc, ATTR_TRANSA, &op_T, @sizeOf(c_int));
        _ = lt.cublasLtMatmulDescSetAttribute.?(opDesc, ATTR_TRANSB, &op_N, @sizeOf(c_int));

        var preference: lt.cublasLtMatmulPreference_t = null;
        if (lt.cublasLtMatmulPreferenceCreate.?(&preference) != lt.CUBLAS_STATUS_SUCCESS) return error.CublasPrefCreateFailed;
        defer _ = lt.cublasLtMatmulPreferenceDestroy.?(preference);
        _ = lt.cublasLtMatmulPreferenceSetAttribute.?(preference, lt.CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size, @sizeOf(usize));

        const elem_size = A_buffer.shape.dtype().sizeOf();
        const A_slice = A_buffer.buffer.slice();
        const B_slice = B_buffer.buffer.slice();
        const C_slice = C_buffer.buffer.slice();

        var off_a: usize = 0;
        var off_b: usize = 0;
        var off_c: usize = 0;

        for (0..group_count) |i| {
            const m: u64 = @intCast(m_dim);
            const k: u64 = @intCast(k_dim);

            const num_tok: c_int = tokens_slice_host_cint[i];

            const a_chunk_size = m * k * elem_size;

            if (num_tok == 0) {
                off_a += a_chunk_size;
                continue;
            }

            const n: u64 = @intCast(num_tok);
            const b_chunk_size = n * k * elem_size;
            const c_chunk_size = m * n * elem_size;

            var Adesc: lt.cublasLtMatrixLayout_t = null;
            var Bdesc: lt.cublasLtMatrixLayout_t = null;
            var Cdesc: lt.cublasLtMatrixLayout_t = null;

            const status_a = lt.cublasLtMatrixLayoutCreate.?(&Adesc, cuda_type, k, m, @intCast(k));
            if (status_a != lt.CUBLAS_STATUS_SUCCESS) return error.CUBLASLtError;

            const status_b = lt.cublasLtMatrixLayoutCreate.?(&Bdesc, cuda_type, k, n, @intCast(k));
            if (status_b != lt.CUBLAS_STATUS_SUCCESS) {
                _ = lt.cublasLtMatrixLayoutDestroy.?(Adesc);
                return error.CUBLASLtError;
            }

            const status_c = lt.cublasLtMatrixLayoutCreate.?(&Cdesc, cuda_type, m, n, @intCast(m));
            if (status_c != lt.CUBLAS_STATUS_SUCCESS) {
                _ = lt.cublasLtMatrixLayoutDestroy.?(Adesc);
                _ = lt.cublasLtMatrixLayoutDestroy.?(Bdesc);
                return error.CUBLASLtError;
            }

            // Heuristic Search
            var heuristicResult: lt.cublasLtMatmulHeuristicResult_t = undefined;
            var returnedResults: c_int = 0;
            const algo_status = lt.cublasLtMatmulAlgoGetHeuristic.?(lt.cublaslt_handle, opDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults);

            if (algo_status != lt.CUBLAS_STATUS_SUCCESS or returnedResults == 0) {
                _ = lt.cublasLtMatrixLayoutDestroy.?(Adesc);
                _ = lt.cublasLtMatrixLayoutDestroy.?(Bdesc);
                _ = lt.cublasLtMatrixLayoutDestroy.?(Cdesc);
                return error.CublasHeuristicFailed;
            }

            // Log per-group sizes and relevant offsets for debugging/profiling.
            // log.info(
            //     "group[{d}] dims: m={d} n={d} k={d} elem_size={d}B offsets(a,b,c)=({d},{d},{d}) chunk_bytes(a,b,c)=({d},{d},{d})",
            //     .{
            //         i,
            //         m,
            //         num_tok,
            //         k,
            //         elem_size,
            //         off_a,
            //         off_b,
            //         off_c,
            //         a_chunk_size,
            //         b_chunk_size,
            //         c_chunk_size,
            //     },
            // );

            const d_A = A_slice[off_a..].ptr;
            const d_B = B_slice[off_b..].ptr;
            const d_C = C_slice[off_c..].ptr;

            // EXECUTION (SWAPPED INPUTS):
            // Arg1 = d_B (Weights), Arg2 = d_A (Activations)
            const run_status = lt.cublasLtMatmul.?(
                lt.cublaslt_handle,
                opDesc,
                &alpha,
                d_A,
                Adesc,
                d_B,
                Bdesc,
                &beta,
                d_C,
                Cdesc,
                d_C,
                Cdesc,
                &heuristicResult.algo,
                device_ws_slice.ptr,
                ws_size,
                cu_stream,
            );

            _ = lt.cublasLtMatrixLayoutDestroy.?(Adesc);
            _ = lt.cublasLtMatrixLayoutDestroy.?(Bdesc);
            _ = lt.cublasLtMatrixLayoutDestroy.?(Cdesc);

            if (run_status != lt.CUBLAS_STATUS_SUCCESS) return error.CublasExecFailed;

            off_a += a_chunk_size;
            off_b += b_chunk_size;
            off_c += c_chunk_size;
        }

        return null;
    }
};

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
    buffer: *const ffi.Buffer,
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

pub fn gemmGroupedBatchedLT(
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
        // computeType: lt.cublasComputeType_t,
        output_shape: zml.Shape,
    },
) [3]Tensor {
    const inputs = .{ A, B, tokens_per_exp, host_buffer, device_buffer };
    return zml.ops.customCall(
        GemmGroupedBatchedLT.custom_call_name,
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
            // .computeType = opts.computeType,
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

    // const memcpyToDeviceAsync = @extern(*const @TypeOf(c.cuMemcpyHtoDAsync_v2), .{ .name = "cuMemcpyHtoDAsync_v2", .linkage = .weak }).?;
    // c.CUdeviceptr = @intCast(@intFromPtr(tokens_per_exp_buffer.ptr.?));

    pub fn memcpyToDeviceAsync(dstDevice: []u8, srcHost: []const u8, stream: c.CUstream) CUresult {
        std.debug.assert(dstDevice.len == srcHost.len);
        const f = @extern(*const @TypeOf(c.cuMemcpyHtoDAsync_v2), .{ .name = "cuMemcpyHtoDAsync_v2", .linkage = .weak }).?;
        return f(@intFromPtr(dstDevice.ptr), @ptrCast(srcHost), srcHost.len, stream);
    }
    // const memcpyToDeviceAsync = @extern(*const @TypeOf(c.cuMemcpyHtoDAsync_v2), .{ .name = "cuMemcpyHtoDAsync_v2", .linkage = .weak }).?;
    // const memcpyToHostAsync = @extern(*const @TypeOf(c.cuMemcpyDtoHAsync_v2), .{ .name = "cuMemcpyDtoHAsync_v2", .linkage = .weak }).?;
    const memcpyToDevice = @extern(*const @TypeOf(c.cuMemcpyHtoD_v2), .{ .name = "cuMemcpyHtoD_v2", .linkage = .weak }).?;

    pub fn memcpyToHostAsync(dstHost: []u8, srcDevice: []const u8, stream: c.CUstream) CUresult {
        std.debug.assert(dstHost.len == srcDevice.len);
        const f = @extern(*const @TypeOf(c.cuMemcpyDtoHAsync_v2), .{ .name = "cuMemcpyDtoHAsync_v2", .linkage = .weak }).?;
        return f(@ptrCast(dstHost), @intFromPtr(srcDevice.ptr), srcDevice.len, stream);
    }

    const memcpyToHost = @extern(*const @TypeOf(c.cuMemcpyDtoH_v2), .{ .name = "cuMemcpyDtoH_v2", .linkage = .weak }).?;
    const streamGetPriority = @extern(*const @TypeOf(c.cuStreamGetPriority), .{ .name = "cuStreamGetPriority", .linkage = .weak }).?;

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
        return Error.CudaError;
    }
};
