const std = @import("std");
const builtin = @import("builtin");

const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const runfiles = @import("runfiles");
const stdx = @import("stdx");

const zml = @import("zml.zig");
const ffi = zml.pjrt.ffi;
const Tensor = zml.Tensor;

const log = std.log.scoped(.@"zml/cublas_grouped_gemm");

// cuBLAS types and constants
pub const cublasStatus_t = c_int;
pub const cublasHandle_t = *anyopaque;
pub const cublasOperation_t = c_int;
pub const cudaDataType_t = c_int;
pub const cublasComputeType_t = c_int;

// 1 cublas datatypes from zml data types
// // -------------------- cublasOperation_t --------------------
pub const CUBLAS_OP_N: cublasOperation_t = 0;
pub const CUBLAS_OP_T: cublasOperation_t = 1;
pub const CUBLAS_OP_C: cublasOperation_t = 2;
pub const CUBLAS_OP_HERMITAN: cublasOperation_t = 2; // synonym of CUBLAS_OP_C
pub const CUBLAS_OP_CONJG: cublasOperation_t = 3;

// -------------------- cublasComputeType_t --------------------
pub const CUBLAS_COMPUTE_16F: cublasComputeType_t = 64;
pub const CUBLAS_COMPUTE_16F_PEDANTIC: cublasComputeType_t = 65;
pub const CUBLAS_COMPUTE_32F: cublasComputeType_t = 68;
pub const CUBLAS_COMPUTE_32F_PEDANTIC: cublasComputeType_t = 69;
pub const CUBLAS_COMPUTE_32F_FAST_16F: cublasComputeType_t = 74;
pub const CUBLAS_COMPUTE_32F_FAST_16BF: cublasComputeType_t = 75;
pub const CUBLAS_COMPUTE_32F_FAST_TF32: cublasComputeType_t = 77;
pub const CUBLAS_COMPUTE_32F_EMULATED_16BFX9: cublasComputeType_t = 78;
pub const CUBLAS_COMPUTE_64F: cublasComputeType_t = 70;
pub const CUBLAS_COMPUTE_64F_PEDANTIC: cublasComputeType_t = 71;
pub const CUBLAS_COMPUTE_32I: cublasComputeType_t = 72;
pub const CUBLAS_COMPUTE_32I_PEDANTIC: cublasComputeType_t = 73;

// -------------------- cudaDataType_t --------------------
pub const CUDA_R_32F: cudaDataType_t = 0;
pub const CUDA_R_64F: cudaDataType_t = 1;
pub const CUDA_R_16F: cudaDataType_t = 2;
pub const CUDA_R_8I: cudaDataType_t = 3;
pub const CUDA_C_32F: cudaDataType_t = 4;
pub const CUDA_C_64F: cudaDataType_t = 5;
pub const CUDA_C_16F: cudaDataType_t = 6;
pub const CUDA_C_8I: cudaDataType_t = 7;
pub const CUDA_R_8U: cudaDataType_t = 8;
pub const CUDA_C_8U: cudaDataType_t = 9;
pub const CUDA_R_32I: cudaDataType_t = 10;
pub const CUDA_C_32I: cudaDataType_t = 11;
pub const CUDA_R_32U: cudaDataType_t = 12;
pub const CUDA_C_32U: cudaDataType_t = 13;
pub const CUDA_R_16BF: cudaDataType_t = 14;
pub const CUDA_C_16BF: cudaDataType_t = 15;
pub const CUDA_R_4I: cudaDataType_t = 16;
pub const CUDA_C_4I: cudaDataType_t = 17;
pub const CUDA_R_4U: cudaDataType_t = 18;
pub const CUDA_C_4U: cudaDataType_t = 19;
pub const CUDA_R_16I: cudaDataType_t = 20;
pub const CUDA_C_16I: cudaDataType_t = 21;
pub const CUDA_R_16U: cudaDataType_t = 22;
pub const CUDA_C_16U: cudaDataType_t = 23;
pub const CUDA_R_64I: cudaDataType_t = 24;
pub const CUDA_C_64I: cudaDataType_t = 25;
pub const CUDA_R_64U: cudaDataType_t = 26;
pub const CUDA_C_64U: cudaDataType_t = 27;
pub const CUDA_R_8F_E4M3: cudaDataType_t = 28;
pub const CUDA_R_8F_UE4M3: cudaDataType_t = 28; // alias
pub const CUDA_R_8F_E5M2: cudaDataType_t = 29;
pub const CUDA_R_8F_UE8M0: cudaDataType_t = 30;
pub const CUDA_R_6F_E2M3: cudaDataType_t = 31;
pub const CUDA_R_6F_E3M2: cudaDataType_t = 32;
pub const CUDA_R_4F_E2M1: cudaDataType_t = 33;

// -------------------- cublasDtatus_t --------------------
pub const CUBLAS_STATUS_SUCCESS: cublasStatus_t = 0;
pub const CUBLAS_STATUS_NOT_INITIALIZED: cublasStatus_t = 1;
pub const CUBLAS_STATUS_ALLOC_FAILED: cublasStatus_t = 3;
pub const CUBLAS_STATUS_INVALID_VALUE: cublasStatus_t = 7;
pub const CUBLAS_STATUS_ARCH_MISMATCH: cublasStatus_t = 8;
pub const CUBLAS_STATUS_MAPPING_ERROR: cublasStatus_t = 11;
pub const CUBLAS_STATUS_EXECUTION_FAILED: cublasStatus_t = 13;
pub const CUBLAS_STATUS_INTERNAL_ERROR: cublasStatus_t = 14;
pub const CUBLAS_STATUS_NOT_SUPPORTED: cublasStatus_t = 15;
pub const CUBLAS_STATUS_LICENSE_ERROR: cublasStatus_t = 16;

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

pub const CublasGemmGroupedBatchedExFunc = *const fn (
    handle: cublasHandle_t,
    transa_array: [*c]const cublasOperation_t,
    transb_array: [*c]const cublasOperation_t,
    m_array: [*c]const c_int,
    n_array: [*c]const c_int,
    k_array: [*c]const c_int,
    alpha_array: ?*const anyopaque,
    Aarray: [*c]const ?*const anyopaque,
    Atype: cudaDataType_t,
    lda_array: [*c]const c_int,
    Barray: [*c]const ?*const anyopaque,
    Btype: cudaDataType_t,
    ldb_array: [*c]const c_int,
    beta_array: ?*const anyopaque,
    Carray: [*c]?*anyopaque,
    Ctype: cudaDataType_t,
    ldc_array: [*c]const c_int,
    group_count: c_int,
    group_size: [*c]const c_int,
    computeType: cublasComputeType_t,
) callconv(.c) cublasStatus_t;

var cublas_gemm_grouped_batched_ex: ?CublasGemmGroupedBatchedExFunc = null;
var cublas_handle: ?cublasHandle_t = null;

fn getScalarAttributeAs(comptime T: type, call_frame: *ffi.CallFrame, attribute_name: []const u8) ?T {
    const attribute = call_frame.attrs.getByName(.scalar, attribute_name) orelse return null;
    return attribute.get(T);
}

fn bufferFromFfiBuffer(ffi_buffer: *const ffi.Buffer) struct { ptr: ?*anyopaque, shape: zml.Shape } {
    const dtype = switch (ffi_buffer.dtype) {
        .f32 => zml.DataType.f32,
        .f16 => zml.DataType.f16,
        .bf16 => zml.DataType.bf16,
        .f64 => zml.DataType.f64,
        .u32 => zml.DataType.u32,
        .i32 => zml.DataType.i32,
        else => unreachable,
    };
    return .{
        .ptr = ffi_buffer.data,
        .shape = zml.Shape.init(ffi_buffer.dims(), dtype),
    };
}

fn getPlatform(call_frame: *ffi.CallFrame) zml.Platform {
    const pjrt_api_ptr = call_frame.attrs.getByName(.scalar, "pjrt_api") orelse unreachable;
    std.debug.assert(pjrt_api_ptr.dtype == .u64);
    const pjrt_api: ?*zml.pjrt.Api = @ptrFromInt(pjrt_api_ptr.get(usize));

    const pjrt_client_ptr = call_frame.attrs.getByName(.scalar, "pjrt_client") orelse unreachable;
    std.debug.assert(pjrt_client_ptr.dtype == .u64);
    const pjrt_client: ?*zml.pjrt.Client = @ptrFromInt(pjrt_client_ptr.get(usize));

    return .{ .target = .cuda, .pjrt_api = pjrt_api.?, .pjrt_client = pjrt_client.? };
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

    fn runInner(call_frame: *ffi.CallFrame) !?*ffi.Error {
        if (call_frame.registeringHook()) return null;

        if (cublas_gemm_grouped_batched_ex == null) {
            return error.CublasNotLoaded;
        }

        if (cublas_handle == null) {
            return error.CublasHandleNotInitialized;
        }

        const num_inputs = call_frame.args.buffers().len;
        stdx.debug.assert(num_inputs >= 2, "Expected at least 2 input buffers (A and B)", .{});

        const group_count: c_int = @intCast(getScalarAttributeAs(i32, call_frame, "group_count") orelse return error.MissingAttribute);
        const alpha: f32 = getScalarAttributeAs(f32, call_frame, "alpha") orelse 1.0;
        const beta: f32 = getScalarAttributeAs(f32, call_frame, "beta") orelse 0.0;
        const computeType: cublasComputeType_t = @intCast(getScalarAttributeAs(i32, call_frame, "computeType") orelse 0);

        // Get arrays from attributes or compute from buffers
        // For simplicity, we'll assume the arrays are passed as attributes
        // In a real implementation, you might need to extract these from buffers

        // For now, we'll create placeholder arrays - you'll need to adapt this
        // based on how you want to pass the parameters
        var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
        defer arena.deinit();

        // Allocate arrays for parameters
        const transa_array = arena.allocator().alloc(cublasOperation_t, @intCast(group_count)) catch unreachable;
        const transb_array = arena.allocator().alloc(cublasOperation_t, @intCast(group_count)) catch unreachable;
        const m_array = arena.allocator().alloc(c_int, @intCast(group_count)) catch unreachable;
        const n_array = arena.allocator().alloc(c_int, @intCast(group_count)) catch unreachable;
        const k_array = arena.allocator().alloc(c_int, @intCast(group_count)) catch unreachable;
        const lda_array = arena.allocator().alloc(c_int, @intCast(group_count)) catch unreachable;
        const ldb_array = arena.allocator().alloc(c_int, @intCast(group_count)) catch unreachable;
        const ldc_array = arena.allocator().alloc(c_int, @intCast(group_count)) catch unreachable;
        const group_size = arena.allocator().alloc(c_int, @intCast(group_count)) catch unreachable;
        const alpha_array = arena.allocator().alloc(f32, @intCast(group_count)) catch unreachable;
        const beta_array = arena.allocator().alloc(f32, @intCast(group_count)) catch unreachable;

        @memset(alpha_array, alpha);
        @memset(beta_array, beta);

        // Get input and output buffers
        const A_buffer = bufferFromFfiBuffer(call_frame.args.buffers()[0]);
        const B_buffer = bufferFromFfiBuffer(call_frame.args.buffers()[1]);
        const tokens_per_exp_buffer = bufferFromFfiBuffer(call_frame.args.buffers()[2]);
        const C_buffer = bufferFromFfiBuffer(call_frame.results.buffers()[0]);

        const stream_ptr = call_frame.stream();
        const cu_stream: c.CUstream = @ptrCast(@constCast(stream_ptr));

        const td = tokens_per_exp_buffer.shape.dims();
        stdx.debug.assert(td.len == 1, "tokens_per_exp expected rank-1, got rank={d}", .{td.len});
        const count: usize = @intCast(td[0]);

        const host_tokens = arena.allocator().alloc(c_int, @intCast(group_count)) catch unreachable;

        stdx.debug.assert(tokens_per_exp_buffer.ptr != null, "tokens_per_exp device ptr is null", .{});
        const src_dev: c.CUdeviceptr = @intCast(@intFromPtr(tokens_per_exp_buffer.ptr.?));

        const bytes: usize = count * @sizeOf(c_int);

        try cuda.check(cuda.memcpyToHostAsync(host_tokens.ptr, src_dev, bytes, cu_stream));

        const Ad = A_buffer.shape.dims();
        const Bd = B_buffer.shape.dims();

        stdx.debug.assert(Ad.len == 2, "Expected A to be rank-2 [m,k], got rank={d}", .{Ad.len});
        stdx.debug.assert(Bd.len == 2, "Expected B to be rank-2 [k,n], got rank={d}", .{Bd.len});

        const elem_size: usize = switch (A_buffer.shape.dtype()) {
            .f16, .bf16, .f32, .f64 => |v| v.sizeOf(),
            else => return error.UnsupportedDataType,
        };

        const Aarray = arena.allocator().alloc(?*const anyopaque, @intCast(group_count)) catch unreachable;
        const Barray = arena.allocator().alloc(?*const anyopaque, @intCast(group_count)) catch unreachable;
        const Carray = arena.allocator().alloc(?*anyopaque, @intCast(group_count)) catch unreachable;

        @memset(Carray, null);

        // Base pointers as bytes for byte-offset arithmetic
        const A_base: [*]const u8 = @ptrCast(@alignCast(A_buffer.ptr));
        const B_base: [*]const u8 = @ptrCast(@alignCast(B_buffer.ptr));
        const C_base: [*]u8 = @ptrCast(@alignCast(C_buffer.ptr));

        var a_off_bytes: usize = 0;
        var b_off_bytes: usize = 0;
        var c_off_bytes: usize = 0;

        const m: c_int = @intCast(Ad[0]);
        const k: c_int = @intCast(Ad[1]);

        // We hard-code: 1 GEMM per group
        for (0..@intCast(group_count)) |gi| {
            transa_array[gi] = CUBLAS_OP_T;
            transb_array[gi] = CUBLAS_OP_N;

            n_array[gi] = host_tokens[gi];
            m_array[gi] = @divExact(m, group_count);
            k_array[gi] = k;

            lda_array[gi] = k;
            ldb_array[gi] = k;
            ldc_array[gi] = @divExact(m, group_count);

            group_size[gi] = 1;

            const lda: usize = @intCast(lda_array[gi]);
            const ldb: usize = @intCast(ldb_array[gi]);
            const ldc: usize = @intCast(ldc_array[gi]);

            const n_g: usize = @intCast(n_array[gi]);
            const m_g: usize = @intCast(m_array[gi]);

            const A_elems: usize = lda * m_g;
            const B_elems: usize = ldb * n_g;
            const C_elems: usize = ldc * n_g;

            Aarray[gi] = @ptrCast(A_base + a_off_bytes);
            Barray[gi] = @ptrCast(B_base + b_off_bytes);
            Carray[gi] = @ptrCast(C_base + c_off_bytes);

            a_off_bytes += A_elems * elem_size;
            b_off_bytes += B_elems * elem_size;
            c_off_bytes += C_elems * elem_size;

            // log.info("G[{d}] A_ptr=0x{x} B_ptr=0x{x} C_ptr=0x{x}", .{ gi, @intFromPtr(Aarray[gi]), @intFromPtr(Barray[gi]), @intFromPtr(Carray[gi]) });

            // log.info("G[{d}] m={d} n={d} k={d} lda={d} ldb={d} ldc={d}", .{
            //     gi, m_array[gi], n_array[gi], k_array[gi], lda, ldb, ldc,
            // });
        }

        // You'll need to populate m_array, n_array, k_array, etc. from your actual data
        const ptr_bytes: usize = @as(usize, @intCast(group_count)) * @sizeOf(?*anyopaque);

        var d_A_array: c.CUdeviceptr = 0;
        var d_B_array: c.CUdeviceptr = 0;
        var d_C_array: c.CUdeviceptr = 0;

        //faire a l'init
        try cuda.check(cuda.memAlloc(&d_A_array, ptr_bytes));
        defer _ = cuda.memFree(d_A_array);

        try cuda.check(cuda.memAlloc(&d_B_array, ptr_bytes));
        defer _ = cuda.memFree(d_B_array);

        try cuda.check(cuda.memAlloc(&d_C_array, ptr_bytes));
        defer _ = cuda.memFree(d_C_array);

        const A_src: ?*const anyopaque = @ptrCast(Aarray.ptr);
        const B_src: ?*const anyopaque = @ptrCast(Barray.ptr);
        const C_src: ?*const anyopaque = @ptrCast(Carray.ptr);

        // Copy host arrays of device pointers -> device arrays
        try cuda.check(cuda.memcpyToDeviceAsync(d_A_array, A_src, ptr_bytes, cu_stream));
        try cuda.check(cuda.memcpyToDeviceAsync(d_B_array, B_src, ptr_bytes, cu_stream));
        try cuda.check(cuda.memcpyToDeviceAsync(d_C_array, C_src, ptr_bytes, cu_stream));

        const Atype: cudaDataType_t = switch (A_buffer.shape.dtype()) {
            .f32 => CUDA_R_32F, // CUDA_R_32F
            .f16 => CUDA_R_16F, // CUDA_R_16F
            .bf16 => CUDA_R_16BF, // CUDA_R_16BF
            .f64 => CUDA_R_64F, // CUDA_R_64F
            else => return error.UnsupportedDataType,
        };
        const Btype: cudaDataType_t = Atype;
        const Ctype: cudaDataType_t = Atype;

        // Call cuBLAS function
        const status = cublas_gemm_grouped_batched_ex.?(
            cublas_handle.?,
            transa_array.ptr,
            transb_array.ptr,
            m_array.ptr,
            n_array.ptr,
            k_array.ptr,
            alpha_array.ptr,
            @ptrFromInt(@as(usize, @intCast(d_A_array))),
            Atype,
            lda_array.ptr,
            @ptrFromInt(@as(usize, @intCast(d_B_array))),
            Btype,
            ldb_array.ptr,
            beta_array.ptr,
            @ptrFromInt(@as(usize, @intCast(d_C_array))),
            Ctype,
            ldc_array.ptr,
            group_count,
            group_size.ptr,
            computeType,
        );

        if (status != CUBLAS_STATUS_SUCCESS) {
            return error.CublasError;
        }

        return null;
    }
};

pub fn load(allocator: std.mem.Allocator) !void {
    _ = allocator; // autofix
    const library_name = if (builtin.os.tag == .macos) "libcublas.dylib" else "libcublas.so";
    const versions = [_][]const u8{ ".13", ".12", ".11", "" };

    for (versions) |version| {
        var name_buf: [256]u8 = undefined;
        const full_name = std.fmt.bufPrintZ(&name_buf, "{s}{s}", .{ library_name, version }) catch continue;

        const handle = std.c.dlopen(full_name, .{ .NOW = true, .GLOBAL = true });
        if (handle) |h| {
            const sym = std.c.dlsym(h, "cublasGemmGroupedBatchedEx");
            if (sym) |s| {
                cublas_gemm_grouped_batched_ex = @ptrCast(@alignCast(s));
                log.info("Successfully loaded cublasGemmGroupedBatchedEx from {s}", .{full_name});

                // Also need to load cublasCreate to initialize handle
                const create_sym = std.c.dlsym(h, "cublasCreate_v2");
                if (create_sym) |create_fn| {
                    const cublasCreate: *const fn (*cublasHandle_t) callconv(.c) cublasStatus_t = @ptrCast(@alignCast(create_fn));
                    var handle_ptr: cublasHandle_t = undefined;
                    const status = cublasCreate(&handle_ptr);
                    if (status == CUBLAS_STATUS_SUCCESS) {
                        cublas_handle = handle_ptr;
                        log.info("Initialized cuBLAS handle", .{});
                    }
                }
                return;
            }
        }
    }

    log.warn("Failed to load libcublas, cublasGemmGroupedBatchedEx will not be available", .{});
    return error.DlOpenFailed;
}

pub fn gemmGroupedBatched(
    A: Tensor,
    B: Tensor,
    tokens_per_exp: Tensor,
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
        computeType: cublasComputeType_t,
        output_shape: zml.Shape,
    },
) Tensor {
    const inputs = .{ A, B, tokens_per_exp };
    return zml.ops.customCall(
        GemmGroupedBatched.custom_call_name,
        inputs,
        .{opts.output_shape},
        .{
            .group_count = opts.group_count,
            .alpha = opts.alpha,
            .beta = opts.beta,
            .computeType = opts.computeType,
            // Note: Arrays need to be passed differently - you might need to
            // create buffers for them or use a different approach
        },
        .{ .has_side_effect = false },
    );
}
const cuda = struct {
    const memcpyToDeviceAsync = @extern(*const @TypeOf(c.cuMemcpyHtoDAsync_v2), .{ .name = "cuMemcpyHtoDAsync_v2", .linkage = .weak }).?;
    const memcpyToHostAsync = @extern(*const @TypeOf(c.cuMemcpyDtoHAsync_v2), .{ .name = "cuMemcpyDtoHAsync_v2", .linkage = .weak }).?;
    const launchHostFunc = @extern(*const @TypeOf(c.cuLaunchHostFunc), .{ .name = "cuLaunchHostFunc", .linkage = .weak }).?;
    const streamSynchronize = @extern(*const @TypeOf(c.cuStreamSynchronize), .{ .name = "cuStreamSynchronize", .linkage = .weak }).?;

    const memAlloc = @extern(*const @TypeOf(c.cuMemAlloc_v2), .{ .name = "cuMemAlloc_v2", .linkage = .weak }).?;
    const memFree = @extern(*const @TypeOf(c.cuMemFree_v2), .{ .name = "cuMemFree_v2", .linkage = .weak }).?;

    pub fn check(result: c.CUresult) error{CudaError}!void {
        if (result == c.CUDA_SUCCESS) return;
        std.log.err("cuda error: {}", .{result});
        return error.CudaError;
    }
};
