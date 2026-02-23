const std = @import("std");

const c = @import("c");
const fused_moe = @import("platforms/cuda/fused_moe");
const platforms = @import("platforms");
const stdx = @import("stdx");

const zml = @import("../zml.zig");
const ffi = zml.pjrt.ffi;
const Tensor = zml.Tensor;

const log = std.log.scoped(.@"zml/fused_moe");

pub fn load(allocator: std.mem.Allocator, io: std.Io) !void {
    _ = allocator; // autofix
    _ = io; // autofix
    if (comptime platforms.isEnabled(.cuda)) {
        // Kernel is statically linked, no need to load
    }
}

pub fn register(platform: zml.Platform) !void {
    if (comptime platforms.isEnabled(.cuda)) {
        try FusedMoE.register(platform);
    }
}

pub const FusedMoE = struct {
    pub const custom_call_name: [:0]const u8 = "fused_moe_kernel";

    pub fn run(call_frame: *ffi.CallFrame) callconv(.c) ?*ffi.Error {
        return runInner(call_frame) catch |err| b: {
            log.err("fused_moe_kernel failed: {}", .{err});
            break :b ffi.Error.create(call_frame.api, .unknown, "fused_moe_kernel failed");
        };
    }

    pub fn register(platform: zml.Platform) !void {
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
        const cu_stream: c.CUstream = @ptrCast(call_frame.api.stream(ctx));

        const buffers = call_frame.args.buffers();
        
        // Extract buffers (all should be on device)
        // Expected order: input, gate_up_blocks, gate_up_scales, [gate_up_bias], down_blocks, down_scales, [down_bias], expert_indices, routing_scores, [token_mask], workspace
        var idx: usize = 0;
        const input_buffer = bufferFromFfiBuffer(buffers[idx]);
        idx += 1;
        const gate_up_blocks_buffer = bufferFromFfiBuffer(buffers[idx]);
        idx += 1;
        const gate_up_scales_buffer = bufferFromFfiBuffer(buffers[idx]);
        idx += 1;
        
        // Check if gate_up_bias is present (non-empty buffer)
        const has_gate_up_bias = idx < buffers.len and buffers[idx].dims().len > 0;
        const gate_up_bias_buffer = if (has_gate_up_bias) bufferFromFfiBuffer(buffers[idx]) else null;
        if (has_gate_up_bias) idx += 1;
        
        const down_blocks_buffer = bufferFromFfiBuffer(buffers[idx]);
        idx += 1;
        const down_scales_buffer = bufferFromFfiBuffer(buffers[idx]);
        idx += 1;
        
        // Check if down_bias is present
        const has_down_bias = idx < buffers.len and buffers[idx].dims().len > 0;
        const down_bias_buffer = if (has_down_bias) bufferFromFfiBuffer(buffers[idx]) else null;
        if (has_down_bias) idx += 1;
        
        const expert_indices_buffer = bufferFromFfiBuffer(buffers[idx]);
        idx += 1;
        const routing_scores_buffer = bufferFromFfiBuffer(buffers[idx]);
        idx += 1;
        
        // Check if token_mask is present
        const has_token_mask = idx < buffers.len and buffers[idx].dims().len > 0;
        const token_mask_buffer = if (has_token_mask) bufferFromFfiBuffer(buffers[idx]) else null;
        if (has_token_mask) idx += 1;
        
        const workspace_buffer = bufferFromFfiBuffer(buffers[idx]);
        
        const output_buffer = bufferFromFfiBuffer(call_frame.results.buffers()[0]);

        const seq_len: c_int = @intCast(input_buffer.shape.dim(0));
        const hidden_dim: c_int = @intCast(input_buffer.shape.dim(1));
        const num_experts: c_int = @intCast(getScalarAttributeAs(c_int, call_frame, "num_experts") orelse return error.MissingNumExperts);
        const top_k: c_int = @intCast(getScalarAttributeAs(c_int, call_frame, "top_k") orelse return error.MissingTopK);
        const ffn_dim: c_int = @intCast(getScalarAttributeAs(c_int, call_frame, "ffn_dim") orelse return error.MissingFfnDim);
        const block_size: c_int = getScalarAttributeAs(c_int, call_frame, "block_size") orelse 1;

        // Get device pointers
        const input_ptr = try input_buffer.buffer._shards.get(0).getOpaqueDeviceMemoryDataPointer(call_frame.api);
        const gate_up_blocks_ptr = try gate_up_blocks_buffer.buffer._shards.get(0).getOpaqueDeviceMemoryDataPointer(call_frame.api);
        const gate_up_scales_ptr = try gate_up_scales_buffer.buffer._shards.get(0).getOpaqueDeviceMemoryDataPointer(call_frame.api);
        const gate_up_bias_ptr = if (gate_up_bias_buffer) |b| try b.buffer._shards.get(0).getOpaqueDeviceMemoryDataPointer(call_frame.api) else null;
        const down_blocks_ptr = try down_blocks_buffer.buffer._shards.get(0).getOpaqueDeviceMemoryDataPointer(call_frame.api);
        const down_scales_ptr = try down_scales_buffer.buffer._shards.get(0).getOpaqueDeviceMemoryDataPointer(call_frame.api);
        const down_bias_ptr = if (down_bias_buffer) |b| try b.buffer._shards.get(0).getOpaqueDeviceMemoryDataPointer(call_frame.api) else null;
        const expert_indices_ptr = try expert_indices_buffer.buffer._shards.get(0).getOpaqueDeviceMemoryDataPointer(call_frame.api);
        const routing_scores_ptr = try routing_scores_buffer.buffer._shards.get(0).getOpaqueDeviceMemoryDataPointer(call_frame.api);
        const token_mask_ptr = if (token_mask_buffer) |b| try b.buffer._shards.get(0).getOpaqueDeviceMemoryDataPointer(call_frame.api) else null;
        const workspace_ptr = try workspace_buffer.buffer._shards.get(0).getOpaqueDeviceMemoryDataPointer(call_frame.api);
        const output_ptr = try output_buffer.buffer._shards.get(0).getOpaqueDeviceMemoryDataPointer(call_frame.api);

        // Launch kernel
        try fused_moe.launchFusedMoEKernel(
            cu_stream,
            input_ptr,
            gate_up_blocks_ptr,
            gate_up_scales_ptr,
            gate_up_bias_ptr,
            down_blocks_ptr,
            down_scales_ptr,
            down_bias_ptr,
            @ptrCast(@alignCast(expert_indices_ptr)),
            @ptrCast(@alignCast(routing_scores_ptr)),
            if (token_mask_ptr) |p| @ptrCast(@alignCast(p)) else null,
            workspace_ptr,
            output_ptr,
            .{
                .seq_len = seq_len,
                .num_experts = num_experts,
                .top_k = top_k,
                .hidden_dim = hidden_dim,
                .ffn_dim = ffn_dim,
                .block_size = block_size,
            },
        );

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
        else => unreachable,
    };
}

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
        .u8 => zml.DataType.u8,
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

pub fn fusedMoEForward(
    input: Tensor,
    gate_up_blocks: Tensor,
    gate_up_scales: Tensor,
    gate_up_bias: ?Tensor,
    down_blocks: Tensor,
    down_scales: Tensor,
    down_bias: ?Tensor,
    expert_indices: Tensor,
    routing_scores: Tensor,
    token_mask: ?Tensor,
    workspace: Tensor,
    opts: struct {
        num_experts: u32,
        top_k: u32,
        ffn_dim: u32,
        block_size: u32 = 1,
        output_shape: zml.Shape,
    },
) Tensor {
    // Build inputs array
    const num_inputs = 7 + (if (gate_up_bias != null) 1 else 0) + (if (down_bias != null) 1 else 0) + (if (token_mask != null) 1 else 0) + 1; // +1 for workspace
    const inputs = zml.module.CompilationContext.current().allocator.alloc(Tensor, num_inputs) catch unreachable;
    var idx: usize = 0;
    inputs[idx] = input;
    idx += 1;
    inputs[idx] = gate_up_blocks;
    idx += 1;
    inputs[idx] = gate_up_scales;
    idx += 1;
    if (gate_up_bias) |b| {
        inputs[idx] = b;
        idx += 1;
    }
    inputs[idx] = down_blocks;
    idx += 1;
    inputs[idx] = down_scales;
    idx += 1;
    if (down_bias) |b| {
        inputs[idx] = b;
        idx += 1;
    }
    inputs[idx] = expert_indices;
    idx += 1;
    inputs[idx] = routing_scores;
    idx += 1;
    if (token_mask) |m| {
        inputs[idx] = m;
        idx += 1;
    }
    inputs[idx] = workspace;
    
    return zml.ops.customCall(
        FusedMoE.custom_call_name,
        inputs,
        .{opts.output_shape},
        .{
            .num_experts = opts.num_experts,
            .top_k = opts.top_k,
            .ffn_dim = opts.ffn_dim,
            .block_size = opts.block_size,
        },
        .{ .has_side_effect = false, .output_operand_aliases = &[_]i64{} },
    );
}
