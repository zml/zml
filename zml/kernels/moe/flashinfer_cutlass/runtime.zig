const std = @import("std");

const fi_cutlass_moe = @import("platforms/cuda/flashinfer_cutlass_moe");
const pjrt = @import("pjrt");

const DataType = @import("../../../dtype.zig").DataType;
const contract = @import("contract.zig");

const log = std.log.scoped(.@"zml/kernels/moe/flashinfer_cutlass/runtime");

// TODO(Corentin): This was coming from Platform but we are now downstream.
// Find a way to solve that.
// Maybe the max_num_devices could be its own module because it only depends on Target.
const max_num_devices = 64;

// TODO(Corentin): replace that with Shape when its dependencies are untangled and
// it doesn't depend on the platform anymore.
fn count(buffer: *const pjrt.ffi.Buffer) usize {
    if (buffer.rank == 0) return @intCast(buffer.dims()[0]);

    var c: usize = 1;
    for (buffer.dims()) |dim| c = c * @as(usize, @intCast(dim));

    return c;
}

fn byteSize(buffer: *const pjrt.ffi.Buffer) usize {
    const dt: DataType = switch (buffer.dtype) {
        .invalid => @panic("Found an invalid pjrt buffer"),
        .token => @panic("Found an token pjrt buffer"),
        inline else => |tag| @field(DataType, @tagName(tag)),
    };
    return std.math.divCeil(usize, count(buffer) * dt.bitSizeOf(), 8) catch unreachable;
}

pub fn handler(call_frame: *pjrt.ffi.CallFrame) callconv(.c) ?*pjrt.ffi.Error {
    if (call_frame.registeringHook()) return null;

    // TODO(Corentin): Find a place to abstract that
    var input_buffers: contract.Inputs(*const pjrt.ffi.Buffer) = undefined;
    inline for (@typeInfo(contract.Inputs(*const pjrt.ffi.Buffer)).@"struct".field_names, 0..) |field_name, i| {
        const buf = call_frame.args.buffers()[i];
        @field(input_buffers, field_name) = buf;
    }

    var output_buffers: contract.Outputs(*const pjrt.ffi.Buffer) = undefined;
    inline for (@typeInfo(contract.Outputs(*const pjrt.ffi.Buffer)).@"struct".field_names, 0..) |field_name, i| {
        const buf = call_frame.results.buffers()[i];
        @field(output_buffers, field_name) = buf;
    }

    const attributes: contract.Attributes = switch (@typeInfo(contract.Attributes)) {
        .void => {},
        .@"struct" => blk: {
            var attrs: contract.Attributes = undefined;
            inline for (@typeInfo(contract.Attributes).@"struct".field_names, @typeInfo(contract.Attributes).@"struct".field_types) |field_name, Field| {
                // TODO: Use getByIndex instead?
                const attribute = call_frame.attrs.getByName(.scalar, field_name) orelse
                    @panic("Attribute not found: " ++ field_name);
                @field(attrs, field_name) = attribute.get(Field);
            }
            break :blk attrs;
        },
        else => @compileError("Unsupported attributes type: " ++ @typeName(contract.Attributes)),
    };

    return handlerInner(call_frame, input_buffers, output_buffers, attributes) catch |err|
        pjrt.ffi.Error.create(call_frame.api, .unknown, @errorName(err));
}

fn handlerInner(
    call_frame: *pjrt.ffi.CallFrame,
    inputs: contract.Inputs(*const pjrt.ffi.Buffer),
    outputs: contract.Outputs(*const pjrt.ffi.Buffer),
    attributes: contract.Attributes,
) !?*pjrt.ffi.Error {
    const device = try call_frame.ctx.getDeviceOrdinal(call_frame.api);
    const runners = try runnersFromAttributes(attributes);
    const deviceRunner = try runners.ensureRunner(device);
    const context = makeContext(attributes);

    var io = std.mem.zeroes(fi_cutlass_moe.Io);
    io.struct_size = @sizeOf(fi_cutlass_moe.Io);
    io.input = inputs.hidden_states.data;
    io.token_selected_experts = @ptrCast(@alignCast(inputs.topk_ids.data));
    io.token_final_scales = @ptrCast(@alignCast(inputs.topk_weights.data));
    io.fc1_expert_weights = inputs.fc1_weights.data;
    io.fc2_expert_weights = inputs.fc2_weights.data;
    io.quant_scales[0] = inputs.fc1_act_global.data;
    io.quant_scales[1] = inputs.fc1_weight_block.data;
    io.quant_scales[2] = inputs.fc1_global.data;
    io.quant_scales[3] = inputs.fc2_act_global.data;
    io.quant_scales[4] = inputs.fc2_weight_block.data;
    io.quant_scales[5] = inputs.fc2_global.data;
    io.quant_scale_count = 6;
    io.quant_scale_per_expert_mask =
        (@as(u32, @intFromBool(attributes.fc1_act_per_expert)) << 0) |
        (@as(u32, @intFromBool(attributes.fc2_act_per_expert)) << 3);
    io.output = outputs.output.data;

    var workspace = std.mem.zeroes(fi_cutlass_moe.Workspace);
    workspace.struct_size = @sizeOf(fi_cutlass_moe.Workspace);
    workspace.data = outputs.workspace.data;
    workspace.data_bytes = byteSize(outputs.workspace);

    try checkStatus(
        deviceRunner.api,
        deviceRunner.api.run(
            deviceRunner.runner,
            &context,
            &io,
            &workspace,
            @ptrCast(call_frame.api.stream(call_frame.ctx)),
            attributes.gemm1_tactic,
            attributes.gemm2_tactic,
        ),
    );
    return null;
}

pub fn makeContext(attributes: contract.Attributes) fi_cutlass_moe.Context {
    var context = std.mem.zeroes(fi_cutlass_moe.Context);
    context.struct_size = @sizeOf(fi_cutlass_moe.Context);
    context.num_tokens = attributes.num_tokens;
    context.hidden_size = attributes.hidden_size;
    context.intermediate_size = attributes.intermediate_size;
    context.num_experts = attributes.num_experts;
    context.num_experts_on_rank = attributes.num_experts;
    context.top_k = attributes.top_k;
    context.tp_size = 1;
    context.ep_size = 1;
    context.activation = @intCast(attributes.activation);
    context.enable_pdl = @intFromBool(attributes.enable_pdl);
    context.swizzled_input_sf = 1;
    return context;
}

pub fn checkStatus(api: *const fi_cutlass_moe.Api, status: fi_cutlass_moe.Status) !void {
    if (status == fi_cutlass_moe.c.ZML_FI_CUTLASS_MOE_STATUS_SUCCESS) return;
    if (api.lastError()) |message| {
        log.err("FlashInfer CUTLASS MoE failed (status {d}): {s}", .{
            status,
            std.mem.span(message),
        });
    } else {
        log.err("FlashInfer CUTLASS MoE failed with status {d}", .{status});
    }
    return switch (status) {
        fi_cutlass_moe.c.ZML_FI_CUTLASS_MOE_STATUS_INVALID_ARGUMENT => error.InvalidArgument,
        fi_cutlass_moe.c.ZML_FI_CUTLASS_MOE_STATUS_UNSUPPORTED => error.UnsupportedArchitecture,
        fi_cutlass_moe.c.ZML_FI_CUTLASS_MOE_STATUS_CUDA_ERROR => error.Cuda,
        else => error.FlashinferCutlassMoe,
    };
}

fn runnersFromAttributes(attributes: contract.Attributes) !*Runners {
    if (attributes.runners == 0) return error.RunnersNotLoaded;
    return @ptrFromInt(attributes.runners);
}

pub const Runners = struct {
    runners: [max_num_devices]?DeviceRunner = @splat(null),

    pub fn deinit(self: *Runners) void {
        for (self.runners) |device_runner| {
            if (device_runner) |runner| {
                _ = runner.api.runnerDestroy(runner.runner);
            }
        }
    }

    pub fn ensureRunner(self: *Runners, device: i32) !DeviceRunner {
        if (device < 0 or device >= max_num_devices) return error.UnsupportedDevice;

        const index: usize = @intCast(device);
        if (self.runners[index]) |runner| return runner;

        const api = try fi_cutlass_moe.apiForDevice(device);
        const options = runnerOptions(device);
        var runner: ?*fi_cutlass_moe.Runner = null;
        try checkStatus(api, api.runnerCreate(&options, &runner));
        const result: DeviceRunner = .{
            .api = api,
            .runner = runner orelse return error.RunnerInitializationFailed,
        };
        self.runners[index] = result;
        return result;
    }
};

const DeviceRunner = struct {
    api: *fi_cutlass_moe.Api,
    runner: *fi_cutlass_moe.Runner,
};

fn runnerOptions(device: i32) fi_cutlass_moe.RunnerOptions {
    var options = std.mem.zeroes(fi_cutlass_moe.RunnerOptions);
    options.struct_size = @sizeOf(fi_cutlass_moe.RunnerOptions);
    options.activation_dtype = fi_cutlass_moe.c.ZML_FI_CUTLASS_MOE_DTYPE_BF16;
    options.weight_dtype = fi_cutlass_moe.c.ZML_FI_CUTLASS_MOE_DTYPE_PACKED_FP4;
    options.output_dtype = fi_cutlass_moe.c.ZML_FI_CUTLASS_MOE_DTYPE_BF16;
    options.device = device;
    options.use_fused_finalize = 1;
    return options;
}

pub fn tacticCounts(
    runners: *const Runners,
    device: i32,
) !struct { gemm1: i32, gemm2: i32 } {
    const deviceRunner = try runners.ensureRunner(device);
    var gemm1: i32 = 0;
    var gemm2: i32 = 0;
    try checkStatus(
        deviceRunner.api,
        deviceRunner.api.getTacticCounts(deviceRunner.runner, &gemm1, &gemm2),
    );
    return .{ .gemm1 = gemm1, .gemm2 = gemm2 };
}

pub fn load(allocator: std.mem.Allocator, io: std.Io, sm: u16) !void {
    try fi_cutlass_moe.load(allocator, io, sm);
}
