const std = @import("std");

const c = @import("c");
const marlin_moe = @import("platforms/cuda/marlin_moe");
const platforms = @import("platforms");
const stdx = @import("stdx");

const zml = @import("../zml.zig");
const ffi = zml.pjrt.ffi;
const Tensor = zml.Tensor;

const log = std.log.scoped(.@"zml/marlin_moe");

pub fn load(allocator: std.mem.Allocator, io: std.Io) !void {
    if (comptime platforms.isEnabled(.cuda)) {
        try marlin_moe.load(allocator, io);
    }
}

pub fn register(platform: *zml.Platform) !void {
    if (comptime platforms.isEnabled(.cuda)) {
        try MarlinMoE.register(platform);
    }
}

pub const MarlinMoE = struct {
    pub const custom_call_name: [:0]const u8 = "marlin_moe_wna16";

    pub fn run(call_frame: *ffi.CallFrame) callconv(.c) ?*ffi.Error {
        return runInner(call_frame) catch |err| b: {
            log.err("marlin_moe_wna16 failed: {}", .{err});
            break :b ffi.Error.create(call_frame.api, .unknown, "marlin_moe_wna16 failed");
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

    fn runInner(call_frame: *ffi.CallFrame) !?*ffi.Error {
        if (call_frame.registeringHook()) return null;

        const ctx: *ffi.ExecutionContext = @constCast(call_frame.ctx);
        const cu_stream: c.CUstream = @ptrCast(call_frame.api.stream(ctx));

        const buffers = call_frame.args.buffers();

        // Extract buffers (all should be on device)
        // Expected order:
        //   a, b_q_weight, b_scales,
        //   [b_zeros], [b_bias], [a_scales], [global_scale], [g_idx], [perm],
        //   sorted_token_ids, expert_ids, num_tokens_past_padded,
        //   [topk_weights], [c_tmp], [a_tmp], [workspace]
        var idx: usize = 0;
        const a_buffer = bufferFromFfiBuffer(buffers[idx]);
        idx += 1;
        const b_q_weight_buffer = bufferFromFfiBuffer(buffers[idx]);
        idx += 1;
        const b_scales_buffer = bufferFromFfiBuffer(buffers[idx]);
        idx += 1;

        const output_buffer = bufferFromFfiBuffer(call_frame.results.buffers()[0]);

        const has_b_zeros = (getScalarAttributeAs(u8, call_frame, "has_b_zeros") orelse 0) != 0;
        const has_bias = (getScalarAttributeAs(u8, call_frame, "has_bias") orelse 0) != 0;
        const has_a_scales = (getScalarAttributeAs(u8, call_frame, "has_a_scales") orelse 0) != 0;
        const has_global_scale = (getScalarAttributeAs(u8, call_frame, "has_global_scale") orelse 0) != 0;
        const has_g_idx = (getScalarAttributeAs(u8, call_frame, "has_g_idx") orelse 0) != 0;
        const has_perm = (getScalarAttributeAs(u8, call_frame, "has_perm") orelse 0) != 0;
        const has_topk_weights = (getScalarAttributeAs(u8, call_frame, "has_topk_weights") orelse 0) != 0;
        const has_c_tmp = (getScalarAttributeAs(u8, call_frame, "has_c_tmp") orelse 0) != 0;
        const has_a_tmp = (getScalarAttributeAs(u8, call_frame, "has_a_tmp") orelse 0) != 0;
        const has_workspace = (getScalarAttributeAs(u8, call_frame, "has_workspace") orelse 0) != 0;

        const b_zeros_buffer = if (has_b_zeros) bufferFromFfiBuffer(buffers[idx]) else null;
        if (has_b_zeros) idx += 1;
        const b_bias_buffer = if (has_bias) bufferFromFfiBuffer(buffers[idx]) else null;
        if (has_bias) idx += 1;
        const a_scales_buffer = if (has_a_scales) bufferFromFfiBuffer(buffers[idx]) else null;
        if (has_a_scales) idx += 1;
        const global_scale_buffer = if (has_global_scale) bufferFromFfiBuffer(buffers[idx]) else null;
        if (has_global_scale) idx += 1;
        const g_idx_buffer = if (has_g_idx) bufferFromFfiBuffer(buffers[idx]) else null;
        if (has_g_idx) idx += 1;
        const perm_buffer = if (has_perm) bufferFromFfiBuffer(buffers[idx]) else null;
        if (has_perm) idx += 1;

        const sorted_token_ids_buffer = bufferFromFfiBuffer(buffers[idx]);
        idx += 1;
        const expert_ids_buffer = bufferFromFfiBuffer(buffers[idx]);
        idx += 1;
        const num_tokens_past_padded_buffer = bufferFromFfiBuffer(buffers[idx]);
        idx += 1;

        const topk_weights_buffer = if (has_topk_weights) bufferFromFfiBuffer(buffers[idx]) else null;
        if (has_topk_weights) idx += 1;
        const c_tmp_buffer = if (has_c_tmp) bufferFromFfiBuffer(buffers[idx]) else null;
        if (has_c_tmp) idx += 1;
        const a_tmp_buffer = if (has_a_tmp) bufferFromFfiBuffer(buffers[idx]) else null;
        if (has_a_tmp) idx += 1;
        const workspace_buffer = if (has_workspace) bufferFromFfiBuffer(buffers[idx]) else null;

        // Get device pointers

        const a_ptr = a_buffer.buffer.data;
        const b_q_weight_ptr = b_q_weight_buffer.buffer.data;
        const b_scales_ptr = b_scales_buffer.buffer.data;
        const output_ptr = output_buffer.buffer.data;
        const b_zeros_ptr = if (b_zeros_buffer) |b| b.buffer.data else null;
        const b_bias_ptr = if (b_bias_buffer) |b| b.buffer.data else null;
        const a_scales_ptr = if (a_scales_buffer) |b| b.buffer.data else null;
        const global_scale_ptr = if (global_scale_buffer) |b| b.buffer.data else null;
        const g_idx_ptr = if (g_idx_buffer) |b| b.buffer.data else null;
        const perm_ptr = if (perm_buffer) |b| b.buffer.data else null;
        const sorted_token_ids_ptr = sorted_token_ids_buffer.buffer.data;
        const expert_ids_ptr = expert_ids_buffer.buffer.data;
        const num_tokens_past_padded_ptr = num_tokens_past_padded_buffer.buffer.data;
        const topk_weights_ptr = if (topk_weights_buffer) |b| b.buffer.data else null;
        const c_tmp_ptr = if (c_tmp_buffer) |b| b.buffer.data else null;
        const a_tmp_ptr = if (a_tmp_buffer) |b| b.buffer.data else null;
        const workspace_ptr = if (workspace_buffer) |b| b.buffer.data else null;
        var params: marlin_moe.Params = std.mem.zeroInit(marlin_moe.Params, .{});
        params.a = a_ptr;
        params.b_q_weight = b_q_weight_ptr;
        params.c = output_ptr;
        params.c_tmp = if (c_tmp_ptr) |p| p else null;
        params.b_bias = if (b_bias_ptr) |p| p else null;
        params.a_scales = if (a_scales_ptr) |p| p else null;
        params.b_scales = b_scales_ptr;
        params.global_scale = if (global_scale_ptr) |p| p else null;
        params.b_zeros = if (b_zeros_ptr) |p| p else null;
        params.g_idx = if (g_idx_ptr) |p| @ptrCast(@alignCast(p)) else null;
        params.perm = if (perm_ptr) |p| @ptrCast(@alignCast(p)) else null;
        params.a_tmp = if (a_tmp_ptr) |p| p else null;
        params.sorted_token_ids = @ptrCast(@alignCast(sorted_token_ids_ptr));
        params.expert_ids = @ptrCast(@alignCast(expert_ids_ptr));
        params.num_tokens_past_padded = @ptrCast(@alignCast(num_tokens_past_padded_ptr));
        params.topk_weights = if (topk_weights_ptr) |p| @ptrCast(@alignCast(p)) else null;
        params.workspace = if (workspace_ptr) |p| p else null;

        params.moe_block_size = getScalarAttributeAs(c_int, call_frame, "moe_block_size") orelse 1;
        params.num_experts = getScalarAttributeAs(c_int, call_frame, "num_experts") orelse return error.MissingNumExperts;
        params.top_k = getScalarAttributeAs(c_int, call_frame, "top_k") orelse return error.MissingTopK;
        params.mul_topk_weights = getScalarAttributeAs(c_int, call_frame, "mul_topk_weights") orelse 0;
        params.size_m = getScalarAttributeAs(c_int, call_frame, "size_m") orelse @intCast(a_buffer.shape.dim(0));
        params.size_n = getScalarAttributeAs(c_int, call_frame, "size_n") orelse return error.MissingSizeN;
        params.size_k = getScalarAttributeAs(c_int, call_frame, "size_k") orelse @intCast(a_buffer.shape.dim(1));
        params.a_type_id = getScalarAttributeAs(i64, call_frame, "a_type_id") orelse return error.MissingATypeId;
        params.b_type_id = getScalarAttributeAs(i64, call_frame, "b_type_id") orelse return error.MissingBTypeId;
        params.c_type_id = getScalarAttributeAs(i64, call_frame, "c_type_id") orelse return error.MissingCTypeId;
        params.s_type_id = getScalarAttributeAs(i64, call_frame, "s_type_id") orelse return error.MissingSTypeId;
        params.has_bias = if (has_bias) 1 else 0;
        params.has_act_order = getScalarAttributeAs(c_int, call_frame, "has_act_order") orelse 0;
        params.is_k_full = getScalarAttributeAs(c_int, call_frame, "is_k_full") orelse 1;
        params.has_zp = getScalarAttributeAs(c_int, call_frame, "has_zp") orelse 0;
        params.num_groups = getScalarAttributeAs(c_int, call_frame, "num_groups") orelse 1;
        params.group_size = getScalarAttributeAs(c_int, call_frame, "group_size") orelse 0;
        params.stream = cu_stream;
        params.thread_k = getScalarAttributeAs(c_int, call_frame, "thread_k") orelse 0;
        params.thread_n = getScalarAttributeAs(c_int, call_frame, "thread_n") orelse 0;
        params.blocks_per_sm = getScalarAttributeAs(c_int, call_frame, "blocks_per_sm") orelse 0;
        params.use_atomic_add = getScalarAttributeAs(c_int, call_frame, "use_atomic_add") orelse 0;
        params.use_fp32_reduce = getScalarAttributeAs(c_int, call_frame, "use_fp32_reduce") orelse 0;
        params.is_zp_float = getScalarAttributeAs(c_int, call_frame, "is_zp_float") orelse 0;

        const status = marlin_moe.marlin_moe_wna16_launch(&params);
        if (status != 0) {
            const c_msg = marlin_moe.marlin_moe_last_error();
            if (c_msg != null and c_msg[0] != 0) {
                log.err("marlin_moe_wna16 failed: {s}", .{std.mem.span(c_msg)});
            } else {
                log.err("marlin_moe_wna16 failed with status {}", .{status});
            }
            return error.KernelFailed;
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

pub fn marlinMoEForward(
    a: Tensor,
    b_q_weight: Tensor,
    b_scales: Tensor,
    b_zeros: ?Tensor,
    b_bias: ?Tensor,
    a_scales: ?Tensor,
    global_scale: ?Tensor,
    g_idx: ?Tensor,
    perm: ?Tensor,
    sorted_token_ids: Tensor,
    expert_ids: Tensor,
    num_tokens_past_padded: Tensor,
    topk_weights: ?Tensor,
    c_tmp: ?Tensor,
    a_tmp: ?Tensor,
    workspace: ?Tensor,
    opts: struct {
        moe_block_size: u32 = 1,
        num_experts: u32,
        top_k: u32,
        mul_topk_weights: u32 = 0,
        size_m: ?u32 = null,
        size_n: u32,
        size_k: ?u32 = null,
        a_type_id: i64,
        b_type_id: i64,
        c_type_id: i64,
        s_type_id: i64,
        has_act_order: u32 = 0,
        is_k_full: u32 = 1,
        has_zp: u32 = 0,
        num_groups: u32 = 1,
        group_size: u32 = 0,
        thread_k: i32 = 0,
        thread_n: i32 = 0,
        blocks_per_sm: i32 = 0,
        use_atomic_add: u32 = 0,
        use_fp32_reduce: u32 = 0,
        is_zp_float: u32 = 0,
        output_shape: zml.Shape,
    },
) Tensor {
    // Build inputs array
    const num_inputs: usize = 6 + // a, b_q_weight, b_scales, sorted_token_ids, expert_ids, num_tokens_past_padded
        @as(usize, @intFromBool(b_zeros != null)) +
        @as(usize, @intFromBool(b_bias != null)) +
        @as(usize, @intFromBool(a_scales != null)) +
        @as(usize, @intFromBool(global_scale != null)) +
        @as(usize, @intFromBool(g_idx != null)) +
        @as(usize, @intFromBool(perm != null)) +
        @as(usize, @intFromBool(topk_weights != null)) +
        @as(usize, @intFromBool(c_tmp != null)) +
        @as(usize, @intFromBool(a_tmp != null)) +
        @as(usize, @intFromBool(workspace != null));
    const inputs = zml.module.CompilationContext.current().allocator.alloc(Tensor, num_inputs) catch unreachable;
    var idx: usize = 0;
    inputs[idx] = a;
    idx += 1;
    inputs[idx] = b_q_weight;
    idx += 1;
    inputs[idx] = b_scales;
    idx += 1;
    if (b_zeros) |b| {
        inputs[idx] = b;
        idx += 1;
    }
    if (b_bias) |b| {
        inputs[idx] = b;
        idx += 1;
    }
    if (a_scales) |b| {
        inputs[idx] = b;
        idx += 1;
    }
    if (global_scale) |b| {
        inputs[idx] = b;
        idx += 1;
    }
    if (g_idx) |b| {
        inputs[idx] = b;
        idx += 1;
    }
    if (perm) |b| {
        inputs[idx] = b;
        idx += 1;
    }
    inputs[idx] = sorted_token_ids;
    idx += 1;
    inputs[idx] = expert_ids;
    idx += 1;
    inputs[idx] = num_tokens_past_padded;
    idx += 1;
    if (topk_weights) |b| {
        inputs[idx] = b;
        idx += 1;
    }
    if (c_tmp) |b| {
        inputs[idx] = b;
        idx += 1;
    }
    if (a_tmp) |b| {
        inputs[idx] = b;
        idx += 1;
    }
    if (workspace) |b| {
        inputs[idx] = b;
    }

    return zml.ops.customCall(
        MarlinMoE.custom_call_name,
        inputs,
        .{opts.output_shape},
        .{
            .has_b_zeros = @as(u8, @intFromBool(b_zeros != null)),
            .has_bias = @as(u8, @intFromBool(b_bias != null)),
            .has_a_scales = @as(u8, @intFromBool(a_scales != null)),
            .has_global_scale = @as(u8, @intFromBool(global_scale != null)),
            .has_g_idx = @as(u8, @intFromBool(g_idx != null)),
            .has_perm = @as(u8, @intFromBool(perm != null)),
            .has_topk_weights = @as(u8, @intFromBool(topk_weights != null)),
            .has_c_tmp = @as(u8, @intFromBool(c_tmp != null)),
            .has_a_tmp = @as(u8, @intFromBool(a_tmp != null)),
            .has_workspace = @as(u8, @intFromBool(workspace != null)),
            .moe_block_size = opts.moe_block_size,
            .num_experts = opts.num_experts,
            .top_k = opts.top_k,
            .mul_topk_weights = opts.mul_topk_weights,
            .size_m = opts.size_m orelse a.shape().dim(0),
            .size_n = opts.size_n,
            .size_k = opts.size_k orelse a.shape().dim(1),
            .a_type_id = opts.a_type_id,
            .b_type_id = opts.b_type_id,
            .c_type_id = opts.c_type_id,
            .s_type_id = opts.s_type_id,
            .has_act_order = opts.has_act_order,
            .is_k_full = opts.is_k_full,
            .has_zp = opts.has_zp,
            .num_groups = opts.num_groups,
            .group_size = opts.group_size,
            .thread_k = opts.thread_k,
            .thread_n = opts.thread_n,
            .blocks_per_sm = opts.blocks_per_sm,
            .use_atomic_add = opts.use_atomic_add,
            .use_fp32_reduce = opts.use_fp32_reduce,
            .is_zp_float = opts.is_zp_float,
        },
        .{ .has_side_effect = false, .output_operand_aliases = &[_]i64{} },
    );
}

pub const FusedMoE = MarlinMoE;
pub const fusedMoEForward = marlinMoEForward;
