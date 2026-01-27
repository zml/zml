const std = @import("std");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const runfiles = @import("runfiles");
const stdx = @import("stdx");

// cuBLAS types and constants
pub const cublasStatus_t = c_int;
pub const cublasHandle_t = *anyopaque;
pub const cublasOperation_t = c_int;
pub const cudaDataType_t = c_int;
pub const cublasComputeType_t = c_int;

// -------------------- cublasOperation_t --------------------
pub const CUBLAS_OP_N: cublasOperation_t = 0;
pub const CUBLAS_OP_T: cublasOperation_t = 1;
pub const CUBLAS_OP_C: cublasOperation_t = 2;
pub const CUBLAS_OP_HERMITAN: cublasOperation_t = 2;
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

pub var cublas_gemm_grouped_batched_ex: ?CublasGemmGroupedBatchedExFunc = null;
pub var cublas_handle: ?cublasHandle_t = null;
pub var cublasSetStream: ?*const fn (cublasHandle_t, ?*anyopaque) callconv(.c) cublasStatus_t = null;
pub var cublasCreate: ?*const fn (*cublasHandle_t) callconv(.c) cublasStatus_t = null;

pub fn load(allocator: std.mem.Allocator, io: std.Io) !void {
    _ = io; // autofix
    _ = allocator; // keep signature like other platform loaders

    // const r = try bazel.runfiles(io, bazel_builtin.current_repository);
    // _ = r; // autofix

    // var buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
    // _ = buffer; // autofix

    // Option A (hermetic): libcublas shipped in runfiles somewhere:
    // const library = (try r.rlocation("PATH/TO/libcublas.so", &buffer)) orelse return error.NotFound;

    // Option B (more consistent with your CUDA runtime): reuse CUDA sandbox layout.
    // In zml/platforms/cuda/cuda.zig the sandbox is:
    //   libpjrt_cuda/sandbox/lib/...
    // So we try the same root here:
    // const sandbox = (try r.rlocation("libpjrt_cuda/sandbox", &buffer)) orelse return error.NotFound;

    // var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    // const library = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox, "lib", "libcublas.so" });

    var lib = std.DynLib.open("libcublas.so") catch |err| {
        std.log.err("Failed to open libcublas: {any} path={s}", .{ err, "libcublas" });
        return err;
    };

    cublas_gemm_grouped_batched_ex =
        lib.lookup(CublasGemmGroupedBatchedExFunc, "cublasGemmGroupedBatchedEx") orelse
        return error.NotFound;

    cublasCreate =
        lib.lookup(*const fn (*cublasHandle_t) callconv(.c) cublasStatus_t, "cublasCreate_v2") orelse
        return error.NotFound;

    cublasSetStream =
        lib.lookup(*const fn (cublasHandle_t, ?*anyopaque) callconv(.c) cublasStatus_t, "cublasSetKernelStream") orelse
        return error.NotFound;

    var handle_ptr: cublasHandle_t = undefined;
    const status = cublasCreate.?(&handle_ptr);
    if (status != CUBLAS_STATUS_SUCCESS) return error.CublasError;

    cublas_handle = handle_ptr;
}

// pub const cuda = struct {
//     pub const memcpyToDeviceAsync = @extern(*const @TypeOf(c.cuMemcpyHtoDAsync_v2), .{ .name = "cuMemcpyHtoDAsync_v2", .linkage = .weak }).?;
//     pub const memcpyToHostAsync = @extern(*const @TypeOf(c.cuMemcpyDtoHAsync_v2), .{ .name = "cuMemcpyDtoHAsync_v2", .linkage = .weak }).?;
//     pub const launchHostFunc = @extern(*const @TypeOf(c.cuLaunchHostFunc), .{ .name = "cuLaunchHostFunc", .linkage = .weak }).?;
//     pub const streamSynchronize = @extern(*const @TypeOf(c.cuStreamSynchronize), .{ .name = "cuStreamSynchronize", .linkage = .weak }).?;

//     pub const memAlloc = @extern(*const @TypeOf(c.cuMemAlloc_v2), .{ .name = "cuMemAlloc_v2", .linkage = .weak }).?;
//     pub const memFree = @extern(*const @TypeOf(c.cuMemFree_v2), .{ .name = "cuMemFree_v2", .linkage = .weak }).?;
//     pub const CUresult = c_int;

//     pub fn check(result: CUresult) error{CudaError}!void {
//         if (result == c.CUDA_SUCCESS) return;
//         std.log.err("cuda error: {}", .{result});
//         return error.CudaError;
//     }
// };
