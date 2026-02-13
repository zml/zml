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

var cublas_mutex: std.Thread.Mutex = .{};
var cublas_handle: ?gg.cublasHandle_t = null;

fn sliceAt(comptime T: type, base: []u8, offset: usize, len: usize) []T {
    const slice: []T = @ptrCast(@alignCast(base[offset..]));
    return slice[0..len];
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

    fn init(base: []u8, layout: anytype, group_count: usize) HostWorkspaceView {
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
            if (group_count == 0 or group_count > 1_000_000) {
                @panic("corrupted group_count");
            }
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

    const InputHostFunc = struct {
        host_workspace_ptr: []u8,
        group_count: usize,
        group_count_cint: c_int,
        out_dim: i64,
        k: i64,
        Atype: zml.DataType,
        A_buffer: []u8,
        B_buffer: []u8,
        C_buffer: []u8,
        alpha: f32,
        beta: f32,
    };

    const HostFunc = struct {
        fn call(userdata: ?*anyopaque) callconv(.c) void {
            const input: *InputHostFunc = @ptrCast(@alignCast(userdata.?));
            // log.info("USERDATA {?*}", .{userdata});

            const group_count = input.group_count;
            const group_count_cint = input.group_count_cint;

            // log.info("group_count {d}", .{group_count});

            // log.info("group_count_cint {d}", .{group_count_cint});

            const A_buffer = input.A_buffer;
            const B_buffer = input.B_buffer;
            const C_buffer = input.C_buffer;

            // log.info(" OUT DIM : {d}", .{input.out_dim});
            const out_dim: c_int = @intCast(input.out_dim);
            const k: c_int = @intCast(input.k);
            // log.info("group count {d}", .{group_count_cint});
            const m = @divExact(out_dim, group_count_cint);

            const layout = MetadataLayout.compute(group_count);
            var ws = HostWorkspaceView.init(input.host_workspace_ptr, layout, group_count); //slices of types

            // Constant metadata for all groups (only `n` varies per group)
            @memset(ws.transa, gg.CUBLAS_OP_T);
            @memset(ws.transb, gg.CUBLAS_OP_N);
            @memset(ws.grp, 1);

            @memset(ws.alpha, input.alpha);
            @memset(ws.beta, input.beta);

            @memset(ws.m, m);
            @memset(ws.k, k);
            @memset(ws.lda, k);
            @memset(ws.ldb, k);
            @memset(ws.ldc, m);

            var a_off: usize = 0;
            var b_off: usize = 0;
            var c_off: usize = 0;

            const elem_size: usize = switch (input.Atype) {
                .f16, .bf16, .f32, .f64 => |v| v.sizeOf(),
                else => @panic("Unsupported datatype"),
            };

            //Use elem size because slicing by type doesn't work because the type is not comptime known

            for (0..group_count) |i| {
                const n: c_int = ws.tokens_copy[i];
                ws.n[i] = n;

                // Store device pointers in the host workspace
                ws.h_ptr_A[i] = @ptrCast(A_buffer[a_off..]);
                // log.info("group_count: {d}, offset: {d}, len={d}, m={d}, n={d}", .{ i, b_off, C_base.len, m, n });

                ws.h_ptr_B[i] = @ptrCast(B_buffer[b_off..]);
                ws.h_ptr_C[i] = @ptrCast(C_buffer[c_off..]);

                // Advance offsets (A: k*m, B: k*n, C: m*n)
                a_off += @as(usize, @intCast(k)) * @as(usize, @intCast(m)) * elem_size;
                b_off += @as(usize, @intCast(k)) * @as(usize, @intCast(n)) * elem_size;
                c_off += @as(usize, @intCast(m)) * @as(usize, @intCast(n)) * elem_size;
            }

            // log.info(
            //     "HostFunc built host workspace: group_count={}, elem_size={}, m={}, k={}, alpha={}, beta={}",
            //     .{ group_count, elem_size, m, k, input.alpha, input.beta },
            // );

            // Dump per-group metadata and the pointer arrays (host memory containing device addresses).
            // for (0..group_count) |i| {
            //     const a_u: usize = if (ws.h_ptr_A[i]) |p| @intFromPtr(p) else 0;
            //     const b_u: usize = if (ws.h_ptr_B[i]) |p| @intFromPtr(p) else 0;
            //     const c_u: usize = if (ws.h_ptr_C[i]) |p| @intFromPtr(p) else 0;

            //     // log.info(
            //     //     "  [g={}] transa={} transb={} m={} n={} k={} lda={} ldb={} ldc={} grp={} alpha={} beta={} tokens_copy={} A=0x{x} B=0x{x} C=0x{x}",
            //     //     .{
            //     //         i,
            //     //         ws.transa[i],
            //     //         ws.transb[i],
            //     //         ws.m[i],
            //     //         ws.n[i],
            //     //         ws.k[i],
            //     //         ws.lda[i],
            //     //         ws.ldb[i],
            //     //         ws.ldc[i],
            //     //         ws.grp[i],
            //     //         ws.alpha[i],
            //     //         ws.beta[i],
            //     //         ws.tokens_copy[i],
            //     //         a_u,
            //     //         b_u,
            //     //         c_u,
            //     //     },
            //     // );
            // }
        }
    };

    fn runInner(call_frame: *ffi.CallFrame) !?*ffi.Error {
        if (call_frame.registeringHook()) return null;

        if (gg.cublas_gemm_grouped_batched_ex == null) {
            return error.CublasNotLoaded;
        }

        // if (gg.cublas_handle == null) {
        //     return error.CublasHandleNotInitialized;
        // }
        if (gg.cublasSetStream == null) {
            return error.CublasSetSreamNotInitialized;
        }

        if (cublas_handle == null) {
            var handle_ptr: gg.cublasHandle_t = undefined;
            const status = gg.cublasCreate.?(&handle_ptr);
            if (status != gg.CUBLAS_STATUS_SUCCESS) {
                return error.CublasInitFailed;
            }
            cublas_handle = handle_ptr;
        }

        const ctx: *ffi.ExecutionContext = @constCast(call_frame.ctx);
        const cu_stream: c.CUstream = @ptrCast(call_frame.api.stream(ctx));

        if (cu_stream == null) {
            log.err("PJRT provided a NULL stream! cuBLAS will default to stream 0.", .{});
        } else {
            // Check if the stream is actually valid according to the driver
            var stream_priority: i32 = 0;
            const res = cuda.streamGetPriority(cu_stream, &stream_priority);
            if (res != c.CUDA_SUCCESS) {
                log.err("Invalid Stream Handle detected! CUDA result: {}", .{res});
                return error.InvalidStream;
            }
        }
        const set_stream_status = gg.cublasSetStream.?(cublas_handle.?, cu_stream);

        if (set_stream_status != gg.CUBLAS_STATUS_SUCCESS) {
            return error.CublasError;
        }

        const buffers = call_frame.args.buffers();

        const A_buffer = bufferFromFfiBuffer(buffers[0]); //

        const B_buffer = bufferFromFfiBuffer(buffers[1]);
        const tokens_per_exp_buffer = bufferFromFfiBuffer(buffers[2]);
        const host_buffer = bufferFromFfiBuffer(buffers[3]);
        const device_buffer = bufferFromFfiBuffer(buffers[4]);

        const group_count_cint: c_int = @intCast(getScalarAttributeAs(c_int, call_frame, "group_count") orelse return error.MissingAttribute);
        const group_count: usize = @intCast(group_count_cint);

        const alpha: f32 = getScalarAttributeAs(f32, call_frame, "alpha") orelse 1.0;
        const beta: f32 = getScalarAttributeAs(f32, call_frame, "beta") orelse 0.0;
        const computeType: gg.cublasComputeType_t = @intCast(getScalarAttributeAs(i32, call_frame, "computeType") orelse 0);

        const C_buffer = bufferFromFfiBuffer(call_frame.results.buffers()[2]);

        const host_buffer_slice = host_buffer.buffer.slice(); // slice of bytes
        const device_buffer_slice = device_buffer.buffer.slice(); // slice of bytes
        _ = device_buffer_slice; // autofix
        const tokens_per_exp_slice = tokens_per_exp_buffer.buffer.slice(); // slice of bytes

        const header_size = std.mem.alignForward(usize, @sizeOf(InputHostFunc), 8);
        if (host_buffer_slice.len < header_size) return error.HostWorkspaceTooSmall;

        const host_workspace_slice = host_buffer_slice[header_size..]; // slice of bytes after header (InputHostFunc)

        const layout = MetadataLayout.compute(group_count);

        const input_ptr = std.mem.bytesAsValue(InputHostFunc, host_buffer_slice);
        input_ptr.* = .{
            .host_workspace_ptr = host_workspace_slice,
            .group_count = group_count,
            .group_count_cint = group_count_cint,
            .out_dim = A_buffer.buffer.dims()[0],
            .k = A_buffer.buffer.dims()[1],
            .Atype = A_buffer.shape.dtype(),
            .A_buffer = A_buffer.buffer.slice(),
            .B_buffer = B_buffer.buffer.slice(),
            .C_buffer = C_buffer.buffer.slice(),
            .alpha = alpha,
            .beta = beta,
        };

        const tokens_copy_slice = host_workspace_slice[layout.offset_tokens_copy..layout.offset_ptr_A_host]; // slice of bytes where write

        try cuda.check(cuda.memcpyToHostAsync(tokens_copy_slice, tokens_per_exp_slice, cu_stream));
        // try cuda.check(cuda.streamSynchronize(cu_stream));

        try cuda.check(cuda.launchHostFunc(cu_stream, HostFunc.call, input_ptr));
        try cuda.check(cuda.ctxSynchronize());
        // try cuda.check(cuda.streamSynchronize(cu_stream));

        const Atype: gg.cudaDataType_t = switch (A_buffer.shape.dtype()) {
            .f32 => gg.CUDA_R_32F,
            .f16 => gg.CUDA_R_16F,
            .bf16 => gg.CUDA_R_16BF,
            .f64 => gg.CUDA_R_64F,
            else => return error.UnsupportedDataType,
        };

        const ws = HostWorkspaceView.init( //return struct of slices of their respective types
            host_workspace_slice,
            layout,
            group_count,
        );

        // const ptr_matrixes_slice = host_workspace_slice[layout.offset_ptr_A_host..layout.total_size];

        // try cuda.check(cuda.memcpyToDeviceAsync(device_buffer_slice, ptr_matrixes_slice, cu_stream));

        // const ptr_array_bytes = std.mem.alignForward(usize, group_count * @sizeOf(?*anyopaque), 16);
        // if (device_buffer_slice.len < 3 * ptr_array_bytes) return error.DeviceWorkspaceTooSmall;

        // const device_a: [*c]const ?*const anyopaque = @ptrCast(@alignCast(device_buffer_slice[0..])); //cast to ptr for kernel
        // _ = device_a; // autofix
        // const device_b: [*c]const ?*const anyopaque = @ptrCast(@alignCast(device_buffer_slice[ptr_array_bytes..]));
        // _ = device_b; // autofix
        // const device_c: [*c]?*anyopaque = @ptrCast(@alignCast(device_buffer_slice[2 * ptr_array_bytes ..]));
        // _ = device_c; // autofix

        // cublas_mutex.lock();

        // const set_stream_status = gg.cublasSetStream.?(gg.cublas_handle.?, cu_stream);

        // try cuda.check(cuda.streamSynchronize(cu_stream));

        const status = gg.cublas_gemm_grouped_batched_ex.?(
            cublas_handle.?,
            ws.transa.ptr,
            ws.transb.ptr,
            ws.m.ptr,
            ws.n.ptr,
            ws.k.ptr,
            ws.alpha.ptr,
            ws.h_ptr_A.ptr,
            // device_a,
            Atype,
            ws.lda.ptr,
            ws.h_ptr_B.ptr,
            // device_b,
            Atype,
            ws.ldb.ptr,
            ws.beta.ptr,
            ws.h_ptr_C.ptr,
            // device_c,
            Atype,
            ws.ldc.ptr,
            group_count_cint,
            ws.grp.ptr,
            computeType,
        );

        // cublas_mutex.unlock();

        if (status != gg.CUBLAS_STATUS_SUCCESS) {
            return error.CublasError;
        }
        try cuda.check(cuda.ctxSynchronize());
        // try cuda.check(cuda.streamSynchronize(cu_stream));

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
