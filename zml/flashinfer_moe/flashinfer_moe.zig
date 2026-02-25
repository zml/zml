const std = @import("std");

const flashinfer_moe_kernel = @import("platforms/cuda/flashinfer_moe_kernel");
const platforms = @import("platforms");

const zml = @import("../zml.zig");
const ffi = zml.pjrt.ffi;
const Tensor = zml.Tensor;

const log = std.log.scoped(.@"zml/flashinfer_moe");

pub fn load(allocator: std.mem.Allocator) !void {
    if (comptime platforms.isEnabled(.cuda)) {
        try flashinfer_moe_kernel.load(allocator);
    }
}

pub fn register(platform: *zml.Platform) !void {
    if (comptime platforms.isEnabled(.cuda)) {
        try FlashinferMoeKernel.register(platform);
    }
}

pub const FlashinferMoeKernel = struct {
    pub const custom_call_name: [:0]const u8 = "flashinfer_moe_kernel";

    pub fn run(call_frame: *ffi.CallFrame) callconv(.c) ?*ffi.Error {
        return runInner(call_frame) catch |err| b: {
            log.err("flashinfer_moe_kernel failed: {}", .{err});
            break :b ffi.Error.create(call_frame.api, .unknown, "flashinfer_moe_kernel failed");
        };
    }

    pub fn register(platform: *zml.Platform) !void {
        try platform.pjrt_api.ffi().?.register(
            platform.pjrt_api,
            custom_call_name,
            "cuda",
            run,
            .{ .command_buffer_compatible = false },
        );
    }

    fn runInner(call_frame: *ffi.CallFrame) !?*ffi.Error {
        if (call_frame.registeringHook()) return null;

        const ctx: *ffi.ExecutionContext = @constCast(call_frame.ctx);
        const cu_stream = call_frame.api.stream(ctx);

        const buffers = call_frame.args.buffers();
        if (buffers.len != 4) return error.InvalidNumInputs;
        if (call_frame.results.buffers().len != 1) return error.InvalidNumOutputs;

        const activations = bufferFromFfiBuffer(buffers[0]);
        const weights_packed = bufferFromFfiBuffer(buffers[1]);
        const scales = bufferFromFfiBuffer(buffers[2]);
        const expert_offsets = bufferFromFfiBuffer(buffers[3]);
        const output = bufferFromFfiBuffer(call_frame.results.buffers()[0]);

        const num_experts: c_int = getScalarAttributeAs(c_int, call_frame, "num_experts") orelse return error.MissingNumExperts;
        const hidden_size: c_int = getScalarAttributeAs(c_int, call_frame, "hidden_size") orelse @intCast(activations.shape.dim(1));
        const out_features: c_int = getScalarAttributeAs(c_int, call_frame, "out_features") orelse @intCast(output.shape.dim(1));

        const status = flashinfer_moe_kernel.moe_cutlass_sm90_bf16_mxfp4_launch_device(
            cu_stream,
            activations.buffer.data,
            @ptrCast(weights_packed.buffer.data),
            @ptrCast(scales.buffer.data),
            @ptrCast(@alignCast(expert_offsets.buffer.data)),
            num_experts,
            hidden_size,
            out_features,
            output.buffer.data,
        );

        if (status != 0) {
            log.err("moe_cutlass_sm90_bf16_mxfp4_launch_device failed with cudaError_t={d}", .{status});
            return error.KernelFailed;
        }

        return null;
    }
};

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
        .u64 => zml.DataType.u64,
        .i32 => zml.DataType.i32,
        .i64 => zml.DataType.i64,
        .i8 => zml.DataType.i8,
        .u8 => zml.DataType.u8,
        .f8e8m0fnu => zml.DataType.f8e8m0,
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

pub fn flashinferMoeForward(
    activations_bf16: Tensor,
    weights_fp4_e2m1_packed: Tensor,
    scales_ue8m0: Tensor,
    expert_first_token_offset_device: Tensor,
    opts: struct {
        num_experts: u32,
        hidden_size: ?u32 = null,
        out_features: u32,
        output_shape: zml.Shape,
    },
) Tensor {
    const inputs = [_]Tensor{
        activations_bf16,
        weights_fp4_e2m1_packed,
        scales_ue8m0,
        expert_first_token_offset_device,
    };
    return zml.ops.customCall(
        FlashinferMoeKernel.custom_call_name,
        @as([]const Tensor, inputs[0..]),
        .{opts.output_shape},
        .{
            .num_experts = opts.num_experts,
            .hidden_size = opts.hidden_size orelse activations_bf16.shape().dim(1),
            .out_features = opts.out_features,
        },
        .{ .has_side_effect = false, .output_operand_aliases = &[_]i64{} },
    );
}
