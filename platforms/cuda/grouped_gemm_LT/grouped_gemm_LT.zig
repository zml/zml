const std = @import("std");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const runfiles = @import("runfiles");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/grouped_gemm/cublaslt");

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

pub const cublasStatus_t = c_int;

pub const cublasLtHandle_t = ?*anyopaque;
pub const cublasLtMatmulDesc_t = ?*anyopaque;
pub const cublasLtMatrixLayout_t = ?*anyopaque;
pub const cublasLtMatmulPreference_t = ?*anyopaque;

pub const cublasLtMatmulAlgo_t = extern struct { data: [8]u64 };

pub const cublasLtMatmulHeuristicResult_t = extern struct {
    algo: cublasLtMatmulAlgo_t,
    workspaceSize: usize,
    state: c_int,
    wavesCount: f32,
    reserved: [4]c_int,
};

// Constants
pub const CUBLAS_COMPUTE_32F = 68;
pub const CUDA_R_16F = 2;
pub const CUDA_R_32F = 0;
pub const CUDA_R_16BF = 14;

// Layout Attributes
pub const CUBLASLT_MATRIX_LAYOUT_ORDER = 3;
pub const CUBLASLT_ORDER_COL = 0;
pub const CUBLASLT_ORDER_ROW = 1;

pub const CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1;

pub const CUBLASLT_MATMUL_DESC_TRANSB: c_int = 0;

pub var cublasLtCreate: ?*const fn (*cublasLtHandle_t) callconv(.c) cublasStatus_t = null;
pub var cublasLtDestroy: ?*const fn (cublasLtHandle_t) callconv(.c) cublasStatus_t = null;

pub var cublasLtMatmulDescCreate: ?*const fn (*cublasLtMatmulDesc_t, c_int, c_int) callconv(.c) cublasStatus_t = null;
pub var cublasLtMatmulDescDestroy: ?*const fn (cublasLtMatmulDesc_t) callconv(.c) cublasStatus_t = null;
pub var cublasLtMatmulDescSetAttribute: ?*const fn (cublasLtMatmulDesc_t, c_int, *const anyopaque, usize) callconv(.c) cublasStatus_t = null;

pub var cublasLtMatrixLayoutCreate: ?*const fn (*cublasLtMatrixLayout_t, c_int, u64, u64, i64) callconv(.c) cublasStatus_t = null;
pub var cublasLtMatrixLayoutDestroy: ?*const fn (cublasLtMatrixLayout_t) callconv(.c) cublasStatus_t = null;
pub var cublasLtMatrixLayoutSetAttribute: ?*const fn (cublasLtMatrixLayout_t, c_int, *const anyopaque, usize) callconv(.c) cublasStatus_t = null;

pub var cublasLtMatmulPreferenceCreate: ?*const fn (*cublasLtMatmulPreference_t) callconv(.c) cublasStatus_t = null;
pub var cublasLtMatmulPreferenceDestroy: ?*const fn (cublasLtMatmulPreference_t) callconv(.c) cublasStatus_t = null;
pub var cublasLtMatmulPreferenceSetAttribute: ?*const fn (cublasLtMatmulPreference_t, c_int, *const anyopaque, usize) callconv(.c) cublasStatus_t = null;

pub var cublasLtMatmulAlgoGetHeuristic: ?*const fn (
    cublasLtHandle_t,
    cublasLtMatmulDesc_t,
    cublasLtMatrixLayout_t, // A
    cublasLtMatrixLayout_t, // B
    cublasLtMatrixLayout_t, // C
    cublasLtMatrixLayout_t, // D
    cublasLtMatmulPreference_t,
    c_int,
    *cublasLtMatmulHeuristicResult_t,
    *c_int,
) callconv(.c) cublasStatus_t = null;

pub var cublasLtMatmul: ?*const fn (
    cublasLtHandle_t,
    cublasLtMatmulDesc_t,
    *const anyopaque, // alpha
    *const anyopaque, // A
    cublasLtMatrixLayout_t, // Adesc
    *const anyopaque, // B
    cublasLtMatrixLayout_t, // Bdesc
    *const anyopaque, // beta
    *const anyopaque, // C
    cublasLtMatrixLayout_t, // Cdesc
    *anyopaque, // D (Output)
    cublasLtMatrixLayout_t, // Ddesc
    *const cublasLtMatmulAlgo_t,
    ?*anyopaque, // Workspace
    usize, // WorkspaceSize
    ?*anyopaque, // Stream
) callconv(.c) cublasStatus_t = null;

pub var cublaslt_handle: cublasLtHandle_t = null;

// The Loader Function
pub fn load(allocator: std.mem.Allocator) !void {
    _ = allocator; // autofix
    // 1. Try to open libcublasLt.so (It is often separate from libcublas.so)
    var lib = std.DynLib.open("libcublas.so") catch |err| {
        // Fallback: On some systems/versions they are bundled in libcublas.so
        std.log.warn("Could not load libcublasLt.so, trying libcublas.so... ({})", .{err});

        return err;
    };

    // 2. Lookup symbols
    cublasLtCreate = lib.lookup(@TypeOf(cublasLtCreate), "cublasLtCreate") orelse return error.SymbolNotFound;
    cublasLtDestroy = lib.lookup(@TypeOf(cublasLtDestroy), "cublasLtDestroy") orelse return error.SymbolNotFound;

    cublasLtMatmulDescCreate = lib.lookup(@TypeOf(cublasLtMatmulDescCreate), "cublasLtMatmulDescCreate") orelse return error.SymbolNotFound;
    cublasLtMatmulDescDestroy = lib.lookup(@TypeOf(cublasLtMatmulDescDestroy), "cublasLtMatmulDescDestroy") orelse return error.SymbolNotFound;
    cublasLtMatmulDescSetAttribute = lib.lookup(@TypeOf(cublasLtMatmulDescSetAttribute), "cublasLtMatmulDescSetAttribute") orelse return error.SymbolNotFound;

    cublasLtMatrixLayoutCreate = lib.lookup(@TypeOf(cublasLtMatrixLayoutCreate), "cublasLtMatrixLayoutCreate") orelse return error.SymbolNotFound;
    cublasLtMatrixLayoutDestroy = lib.lookup(@TypeOf(cublasLtMatrixLayoutDestroy), "cublasLtMatrixLayoutDestroy") orelse return error.SymbolNotFound;
    cublasLtMatrixLayoutSetAttribute = lib.lookup(@TypeOf(cublasLtMatrixLayoutSetAttribute), "cublasLtMatrixLayoutSetAttribute") orelse return error.SymbolNotFound;

    cublasLtMatmulPreferenceCreate = lib.lookup(@TypeOf(cublasLtMatmulPreferenceCreate), "cublasLtMatmulPreferenceCreate") orelse return error.SymbolNotFound;
    cublasLtMatmulPreferenceDestroy = lib.lookup(@TypeOf(cublasLtMatmulPreferenceDestroy), "cublasLtMatmulPreferenceDestroy") orelse return error.SymbolNotFound;
    cublasLtMatmulPreferenceSetAttribute = lib.lookup(@TypeOf(cublasLtMatmulPreferenceSetAttribute), "cublasLtMatmulPreferenceSetAttribute") orelse return error.SymbolNotFound;

    cublasLtMatmulAlgoGetHeuristic = lib.lookup(@TypeOf(cublasLtMatmulAlgoGetHeuristic), "cublasLtMatmulAlgoGetHeuristic") orelse return error.SymbolNotFound;
    cublasLtMatmul = lib.lookup(@TypeOf(cublasLtMatmul), "cublasLtMatmul") orelse return error.SymbolNotFound;

    // 3. Initialize the global handle
    var handle_ptr: cublasLtHandle_t = null;
    const status = cublasLtCreate.?(&handle_ptr);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std.log.err("cublasLtCreate failed with status: {}", .{status});
        return error.CublasInitFailed;
    }
    cublaslt_handle = handle_ptr;

    // Note: leaking 'lib' here is standard for global library loading (it lives until process exit)
    // If you need to unload, you must store 'lib' in a global variable.
}
