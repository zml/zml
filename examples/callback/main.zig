const std = @import("std");

const asynk = @import("async");
const runtimes = @import("runtimes");
const zml = @import("zml");
const cu = zml.platform_specific;

pub const std_options: std.Options = .{
    .log_level = .info,
    .logFn = asynk.logFn(std.log.defaultLog),
};

const log = std.log.scoped(.@"examples/custom_call");

/// Demonstration of the custom_call mechanism.
///
/// * defines a function to compute the grayscale version of an image.
///     * using simple for loop on CPU
///     * using a custom PTX kernel on Cuda, with manual cuLaunchKernel
/// * Use `zml.customCall` to create an executable calling our callback and our kernel.
pub const GrayScale = struct {
    // Mandatory fields to work with ZML custom call api
    pub var type_id: zml.pjrt.ffi.TypeId = undefined;
    pub const callback_config: zml.callback.Config = .{};
    platform: zml.Platform,

    // Custom field to store our cuda module and function.
    cu_data: [2]*anyopaque,

    // Set by ZML before `call` is entered
    results: [1]zml.Buffer = undefined,
    stream: *zml.pjrt.Stream = undefined,

    pub fn init(platform: zml.Platform) !GrayScale {
        var cu_data: [2]*anyopaque = undefined;
        if (comptime runtimes.isEnabled(.cuda)) {
            var module: cu.CUmodule = undefined;
            try cuda.check(cuda.moduleLoadData.?(&module, grayscale_ptx));
            log.info("Loaded Grayscale cuda module", .{});
            var function: cu.CUfunction = undefined;
            try cuda.check(cuda.moduleGetFunction.?(&function, module, "rgba_to_grayscale"));
            log.info("Found Grayscale cuda function", .{});
            cu_data = .{ module.?, function.? };
        }
        return .{
            .platform = platform,
            .cu_data = cu_data,
        };
    }

    pub fn deinit(self: *GrayScale) void {
        if (comptime runtimes.isEnabled(.cuda)) {
            const module: cu.CUmodule = @ptrCast(self.cu_data[0]);
            _ = cuda.moduleUnload.?(module);
        }
    }

    pub fn call(self: *GrayScale, rgb_d: zml.Buffer) !void {
        switch (self.platform.target) {
            .cpu => grayScaleCpu(rgb_d, self.results[0]),
            // Only try to compile `grayScaleCuda` if we have cuda symbols.
            .cuda => if (comptime runtimes.isEnabled(.cuda))
                try self.grayScaleCuda(rgb_d, self.results[0])
            else
                unreachable,
            else => @panic("Platform not supported"),
        }
    }

    pub fn grayScaleCpu(rgb_d: zml.Buffer, gray_d: zml.Buffer) void {
        const rgb_h = rgb_d.asHostBuffer().items(u8);
        const gray_h = gray_d.asHostBuffer().mutItems(u8);

        for (gray_h, 0..) |*gray, i| {
            const px = rgb_h[i * 3 .. i * 3 + 3];
            const R: u32 = @intCast(px[0]);
            const G: u32 = @intCast(px[1]);
            const B: u32 = @intCast(px[2]);
            const gray_u32: u32 = @divFloor(299 * R + 587 * G + 114 * B, 1000);
            gray.* = @intCast(gray_u32);
        }
    }

    pub fn grayScaleCuda(self: GrayScale, rgb_d: zml.Buffer, gray_d: zml.Buffer) !void {
        var args: [2][]u8 = .{
            rgb_d.opaqueDeviceMemoryDataPointer()[0..rgb_d.shape().byteSize()],
            gray_d.opaqueDeviceMemoryDataPointer()[0..gray_d.shape().byteSize()],
        };
        var args_ptr: [2:null]?*anyopaque = .{ @ptrCast(&args[0]), @ptrCast(&args[1]) };
        // This is a naive kernel with one block per pixel.
        try cuda.check(cuda.launchKernel.?(
            @ptrCast(self.cu_data[1]), // function
            @intCast(rgb_d.shape().count() / 3), // num blocks x
            1, // num blocks y
            1, // num blocks z
            1, // num grids x
            1, // num grids y
            1, // num grids z
            0, // shared mem
            @ptrCast(self.stream),
            &args_ptr,
            null,
        ));
        // Note: no explicit synchronization, we just enqueue work in the stream.
    }

    const cuda = struct {
        // Here we leverage ZML sandboxing to access cuda symbols and their definitions.
        const moduleLoadData = @extern(*const @TypeOf(cu.cuModuleLoadData), .{ .name = "cuModuleLoadData", .linkage = .weak });
        const moduleUnload = @extern(*const @TypeOf(cu.cuModuleUnload), .{ .name = "cuModuleUnload", .linkage = .weak });
        const moduleGetFunction = @extern(*const @TypeOf(cu.cuModuleGetFunction), .{ .name = "cuModuleGetFunction", .linkage = .weak });
        const launchKernel = @extern(*const @TypeOf(cu.cuLaunchKernel), .{ .name = "cuLaunchKernel", .linkage = .weak });

        pub fn check(result: cu.CUresult) error{CudaError}!void {
            if (result == cu.CUDA_SUCCESS) return;
            std.log.err("cuda error: {}", .{result});
            return error.CudaError;
        }
    };
};

pub fn grayscale(rgb: zml.Tensor) zml.Tensor {
    const gray_shape = rgb.shape().setDim(0, @divExact(rgb.dim(0), 3));
    const result = zml.callback.call(GrayScale, .{rgb.print()}, &.{gray_shape});
    return result[0];
}

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.smp_allocator, asyncMain);
}

pub fn asyncMain() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var context = try zml.Context.init();
    defer context.deinit();

    const platform = context.autoPlatform(.{});
    context.printAvailablePlatforms(platform);

    // Register our custom call
    try platform.registerCallback(GrayScale);

    // Compile MLIR code containing our custom call.
    const rgb_shape = zml.Shape.init(.{12 * 3}, .u8);
    const exe = try zml.compileFn(allocator, grayscale, .{rgb_shape}, platform);
    defer exe.deinit();

    // Provide runtime information needed by our custom call.
    var gray_op: GrayScale = try .init(platform);
    defer gray_op.deinit();
    try exe.bind(GrayScale, &gray_op);

    // Load data and run the executable.
    const rgb_h: [12][3]u8 = @splat(.{ 0xFF, 0xAA, 0x00 });
    const rgb_d = try zml.Buffer.fromBytes(platform, rgb_shape, @ptrCast(&rgb_h));
    defer rgb_d.deinit();

    var gray_d: zml.Buffer = exe.call(.{rgb_d});
    defer gray_d.deinit();

    // Inspect results
    const gray_h = try gray_d.toHostAlloc(allocator);
    defer gray_h.deinit(allocator);

    std.debug.print("Grayscale conversion of {any} -> {d}\n", .{ rgb_h, gray_h });
    try std.testing.expectEqualSlices(u8, &@as([12]u8, @splat(0xB0)), gray_h.items(u8));
}

// Compiled with Zig and a little help from `https://github.com/gwenzek/cudaz`
const grayscale_ptx =
    \\.version 4.0
    \\.target sm_32
    \\.address_size 64
    \\
    \\.visible .entry rgba_to_grayscale(
    \\    .param .align 8 .b8 rgba_to_grayscale_param_0[16],
    \\    .param .align 8 .b8 rgba_to_grayscale_param_1[16]
    \\)
    \\{
    \\    .reg .pred  %p<2>;
    \\    .reg .b16   %rs<4>;
    \\    .reg .b32   %r<13>;
    \\    .reg .b64   %rd<9>;
    \\
    \\    ld.param.u64    %rd5, [rgba_to_grayscale_param_1+8];
    \\    mov.u32     %r1, %tid.x;
    \\    mov.u32     %r3, %ntid.x;
    \\    mov.u32     %r2, %ctaid.x;
    \\    mad.lo.s32  %r4, %r2, %r3, %r1;
    \\    cvt.u64.u32     %rd1, %r4;
    \\    setp.gt.u64     %p1, %rd5, %rd1;
    \\    @%p1 bra    $L__BB0_2;
    \\    bra.uni     $L__BB0_1;
    \\$L__BB0_2:
    \\    ld.param.u64    %rd4, [rgba_to_grayscale_param_1];
    \\    ld.param.u64    %rd2, [rgba_to_grayscale_param_0];
    \\    cvt.u32.u64     %r5, %rd1;
    \\    mul.lo.s32  %r6, %r5, 3;
    \\    cvt.u64.u32     %rd6, %r6;
    \\    add.s64     %rd7, %rd2, %rd6;
    \\    ld.u8   %rs1, [%rd7];
    \\    ld.u8   %rs2, [%rd7+1];
    \\    ld.u8   %rs3, [%rd7+2];
    \\    mul.wide.u16    %r7, %rs1, 299;
    \\    mul.wide.u16    %r8, %rs2, 587;
    \\    add.s32     %r9, %r8, %r7;
    \\    mul.wide.u16    %r10, %rs3, 114;
    \\    add.s32     %r11, %r9, %r10;
    \\    mul.hi.u32  %r12, %r11, 4294968;
    \\    add.s64     %rd8, %rd4, %rd1;
    \\    st.u8   [%rd8], %r12;
    \\$L__BB0_1:
    \\    ret;
    \\}
;
