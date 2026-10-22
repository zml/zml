const std = @import("std");
const stdx = @import("stdx");

const fi_cutlass_moe = @import("platforms/cuda/flashinfer_cutlass_moe");
const platforms = @import("platforms");
const zml = @import("../../../zml.zig");

const log = std.log.scoped(.@"zml/moe/cutlass_flashinfer");

pub const auto_tactic: i32 = -1;

pub const Options = struct {
    /// Device used at graph-construction time to select the architecture
    /// library and size the XLA-owned scratch buffer.
    workspace_query_device: i32,
    activation: zml.moe.Activation,
    enable_pdl: bool,
    gemm1_tactic: i32,
    /// GEMM2 uses FlashInfer's absolute tactic index. Query tacticCounts() to
    /// obtain the first valid GEMM2 index.
    gemm2_tactic: i32,

    pub fn fromGlobalOptions(opts: zml.moe.Options) Options {
        return .{
            .workspace_query_device = 0,
            .activation = opts.activation,
            .enable_pdl = false,
            .gemm1_tactic = auto_tactic,
            .gemm2_tactic = auto_tactic,
        };
    }
};

pub fn validateOptions(opts: zml.moe.Options) void {
    stdx.debug.assert(!opts.quantize_input, "Optional FP8 input quantization requires the Triton MoE backend", .{});
    stdx.debug.assert(opts.routing_weight_placement == .after_down, "Non-Triton MoE backends require routing weights after the down projection", .{});
    stdx.debug.assert(opts.activation == .swiglu, "flashinfer_cutlass backend only accepts swiglu activation, got {}", .{opts.activation});
    stdx.debug.assert(opts.activation.swiglu.limit == null, "Activation thresholds require the Triton MoE backend", .{});
    stdx.debug.assert(opts.activation.swiglu.bias == null, "flashinfer_cutlass backend requires swiglu bias to be null", .{});
    stdx.debug.assert(opts.activation.swiglu.scale == null, "flashinfer_cutlass backend requires swiglu scale to be null", .{});
}

const Input = struct {
    hidden_states: zml.Tensor,
    fc1_weights: zml.Tensor,
    fc2_weights: zml.Tensor,
    topk_weights: zml.Tensor,
    topk_ids: zml.Tensor,
    fc1_act_global: zml.Tensor,
    fc1_weight_block: zml.Tensor,
    fc1_global: zml.Tensor,
    fc2_act_global: zml.Tensor,
    fc2_weight_block: zml.Tensor,
    fc2_global: zml.Tensor,
};

const Output = struct {
    output: zml.Shape,
    workspace: zml.Shape,
};

const Attributes = struct {
    runners: u64 = 0,
    num_tokens: i64,
    hidden_size: i64,
    intermediate_size: i64,
    num_experts: i32,
    top_k: i32,
    activation: i32,
    enable_pdl: bool,
    fc1_act_per_expert: bool,
    fc2_act_per_expert: bool,
    gemm1_tactic: i32,
    gemm2_tactic: i32,
};

const DeviceRunner = struct {
    api: *fi_cutlass_moe.Api,
    runner: *fi_cutlass_moe.Runner,
};

const max_num_devices = zml.platform.Platform.MAX_NUM_DEVICES;

pub const Runners = struct {
    runners: [max_num_devices]?DeviceRunner = @splat(null),

    pub fn deinit(self: *Runners) void {
        for (self.runners) |device_runner| {
            if (device_runner) |runner| {
                _ = runner.api.runnerDestroy(runner.runner);
            }
        }
    }

    fn ensureRunner(self: *Runners, device: i32) !DeviceRunner {
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

fn checkStatus(api: *const fi_cutlass_moe.Api, status: fi_cutlass_moe.Status) !void {
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

fn runnersFromAttributes(attributes: Attributes) !*Runners {
    if (attributes.runners == 0) return error.RunnersNotLoaded;
    return @ptrFromInt(attributes.runners);
}

fn currentRunners() !*Runners {
    const platform = zml.Compiler.current().platform;
    const runners = platform.state.cuda.fi_cutlass_moe_runners;
    return runners orelse error.RunnersNotLoaded;
}

fn cutlassActivation(activation: zml.moe.Activation) i32 {
    return switch (activation) {
        .gelu => fi_cutlass_moe.c.ZML_FI_CUTLASS_MOE_ACTIVATION_GELU,
        .relu => fi_cutlass_moe.c.ZML_FI_CUTLASS_MOE_ACTIVATION_RELU,
        .silu => fi_cutlass_moe.c.ZML_FI_CUTLASS_MOE_ACTIVATION_SILU,
        .swiglu => fi_cutlass_moe.c.ZML_FI_CUTLASS_MOE_ACTIVATION_SWIGLU,
        .swiglu_step => fi_cutlass_moe.c.ZML_FI_CUTLASS_MOE_ACTIVATION_SWIGLU_STEP,
        .geglu => fi_cutlass_moe.c.ZML_FI_CUTLASS_MOE_ACTIVATION_GEGLU,
        .geglu_tanh => fi_cutlass_moe.c.ZML_FI_CUTLASS_MOE_ACTIVATION_GEGLU_TANH,
    };
}

fn makeContext(attributes: Attributes) fi_cutlass_moe.Context {
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

fn ffiCallNvfp4(
    call_frame: *zml.pjrt.ffi.CallFrame,
    input: zml.pjrtx.TensorToCustomCallBuffer(Input),
    output: zml.pjrtx.ShapeToCustomCallBuffer(Output),
    attributes: Attributes,
) !?*zml.pjrt.ffi.Error {
    const device = try call_frame.ctx.getDeviceOrdinal(call_frame.api);
    const runners = try runnersFromAttributes(attributes);
    const deviceRunner = try runners.ensureRunner(device);
    const context = makeContext(attributes);

    var io = std.mem.zeroes(fi_cutlass_moe.Io);
    io.struct_size = @sizeOf(fi_cutlass_moe.Io);
    io.input = input.hidden_states.ptr;
    io.token_selected_experts = @ptrCast(@alignCast(input.topk_ids.ptr));
    io.token_final_scales = @ptrCast(@alignCast(input.topk_weights.ptr));
    io.fc1_expert_weights = input.fc1_weights.ptr;
    io.fc2_expert_weights = input.fc2_weights.ptr;
    io.quant_scales[0] = input.fc1_act_global.ptr;
    io.quant_scales[1] = input.fc1_weight_block.ptr;
    io.quant_scales[2] = input.fc1_global.ptr;
    io.quant_scales[3] = input.fc2_act_global.ptr;
    io.quant_scales[4] = input.fc2_weight_block.ptr;
    io.quant_scales[5] = input.fc2_global.ptr;
    io.quant_scale_count = 6;
    io.quant_scale_per_expert_mask =
        (@as(u32, @intFromBool(attributes.fc1_act_per_expert)) << 0) |
        (@as(u32, @intFromBool(attributes.fc2_act_per_expert)) << 3);
    io.output = output.output.ptr;

    var workspace = std.mem.zeroes(fi_cutlass_moe.Workspace);
    workspace.struct_size = @sizeOf(fi_cutlass_moe.Workspace);
    workspace.data = output.workspace.ptr;
    workspace.data_bytes = output.workspace.shape.byteSize();

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

const routedNvfp4Call = zml.ops.CustomCall(Input, Output, Attributes, ffiCallNvfp4, .{
    .name = "flashinfer_cutlass_nvfp4_routed_moe",
    // Expert mesh is owned by forwardMoe outer manual computation.
    .sharding_aware = false,
    .has_side_effect = false,
});

fn computeCapability(platform: *const zml.Platform) !u16 {
    const cc = zml.platform.cuda.computeCapability(platform) orelse return error.UnsupportedPlatform;
    return switch (cc.sm()) {
        100, 103, 120 => |sm| sm,
        else => error.UnsupportedArchitecture,
    };
}

pub fn load(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
) !void {
    if (comptime platforms.isEnabled(.cuda)) {
        var cuda_state = &platform.state.cuda;
        if (cuda_state.fi_cutlass_moe_runners != null) return;
        try fi_cutlass_moe.load(allocator, io, try computeCapability(platform));
        const runners = try allocator.create(Runners);
        runners.* = .{};
        cuda_state.fi_cutlass_moe_runners = runners;
        return;
    }
    return error.UnsupportedPlatform;
}

pub fn register(platform: *const zml.Platform) !void {
    if (comptime platforms.isEnabled(.cuda)) {
        try routedNvfp4Call.register(platform);
        return;
    }
    return error.UnsupportedPlatform;
}

pub fn isAvailable(platform: *const zml.Platform) bool {
    if (platform.target != .cuda) return false;
    if (platform.state.cuda.fi_cutlass_moe_runners == null) return false;
    _ = computeCapability(platform) catch return false;
    return true;
}

pub fn isNvfp4Supported(platform: *const zml.Platform) bool {
    if (platform.state.cuda.fi_cutlass_moe_runners == null) return false;
    const compute_capability = computeCapability(platform) catch return false;
    return switch (compute_capability) {
        100, 103, 120 => true,
        else => false,
    };
}

pub fn tacticCounts(
    platform: *const zml.Platform,
    device: i32,
) !struct { gemm1: i32, gemm2: i32 } {
    const runners = platform.state.cuda.fi_cutlass_moe_runners orelse return error.RunnersNotLoaded;
    const deviceRunner = try runners.ensureRunner(device);
    var gemm1: i32 = 0;
    var gemm2: i32 = 0;
    try checkStatus(
        deviceRunner.api,
        deviceRunner.api.getTacticCounts(deviceRunner.runner, &gemm1, &gemm2),
    );
    return .{ .gemm1 = gemm1, .gemm2 = gemm2 };
}

fn roundUp(value: i64, alignment: i64) i64 {
    return @divTrunc(value + alignment - 1, alignment) * alignment;
}

pub fn fc1BlockScaleShape(
    num_experts: i64,
    hidden_size: i64,
    intermediate_size: i64,
    activation: zml.moe.Activation,
) zml.Shape {
    const rows = switch (activation) {
        .silu, .gelu, .relu => intermediate_size,
        .swiglu, .geglu, .geglu_tanh, .swiglu_step => 2 * intermediate_size,
    };
    return .init(
        .{ num_experts, roundUp(rows, 128), roundUp(@divExact(hidden_size, 16), 4) },
        .f8e4m3fn,
    );
}

pub fn fc2BlockScaleShape(
    num_experts: i64,
    hidden_size: i64,
    intermediate_size: i64,
) zml.Shape {
    return .init(
        .{ num_experts, roundUp(hidden_size, 128), roundUp(@divExact(intermediate_size, 16), 4) },
        .f8e4m3fn,
    );
}

fn isGlobalOrPerExpertScale(tensor: zml.Tensor, num_experts: i64) bool {
    return tensor.dtype() == .f32 and
        (tensor.rank() == 0 or (tensor.rank() == 1 and tensor.dim(0) == num_experts));
}

fn validateInputs(
    hidden_states: zml.Tensor,
    fc1_weights: zml.Tensor,
    fc2_weights: zml.Tensor,
    topk_weights: zml.Tensor,
    topk_ids: zml.Tensor,
    fc1_act_global: zml.Tensor,
    fc1_weight_block: zml.Tensor,
    fc1_global: zml.Tensor,
    fc2_act_global: zml.Tensor,
    fc2_weight_block: zml.Tensor,
    fc2_global: zml.Tensor,
    options: Options,
) !Attributes {
    if (hidden_states.dtype() != .bf16 or
        fc1_weights.dtype() != .f4e2m1 or
        fc2_weights.dtype() != .f4e2m1 or
        topk_weights.dtype() != .f32 or
        topk_ids.dtype() != .i32 or
        fc1_weight_block.dtype() != .f8e4m3fn or
        fc2_weight_block.dtype() != .f8e4m3fn or
        fc1_global.dtype() != .f32 or
        fc2_global.dtype() != .f32)
    {
        return error.UnsupportedType;
    }
    if (hidden_states.rank() != 2 or
        fc1_weights.rank() != 3 or
        fc2_weights.rank() != 3 or
        topk_weights.rank() != 2 or
        topk_ids.rank() != 2)
    {
        return error.InvalidShape;
    }

    const tokens = hidden_states.dim(.t);
    const hidden_size = hidden_states.dim(.d);
    const num_experts = fc1_weights.dim(0);
    const fc1_rows = fc1_weights.dim(1);
    const intermediate_size = switch (options.activation) {
        .silu, .gelu, .relu => fc1_rows,
        .swiglu, .geglu, .geglu_tanh, .swiglu_step => @divExact(fc1_rows, 2),
    };
    const top_k = topk_ids.dim(.topk);

    if (tokens <= 0 or hidden_size <= 0 or
        num_experts <= 0 or intermediate_size <= 0 or
        @mod(hidden_size, 16) != 0 or @mod(intermediate_size, 16) != 0 or
        top_k <= 0 or top_k > num_experts)
    {
        return error.InvalidShape;
    }
    if (fc1_weights.dim(2) != hidden_size or
        fc2_weights.dim(0) != num_experts or
        fc2_weights.dim(1) != hidden_size or
        fc2_weights.dim(2) != intermediate_size or
        topk_weights.dim(.t) != tokens or
        topk_weights.dim(.topk) != top_k or
        topk_ids.dim(.t) != tokens or
        !fc1_weight_block.shape().eql(fc1BlockScaleShape(
            num_experts,
            hidden_size,
            intermediate_size,
            options.activation,
        )) or
        !fc2_weight_block.shape().eql(fc2BlockScaleShape(
            num_experts,
            hidden_size,
            intermediate_size,
        )) or
        !fc1_global.shape().eql(.init(.{num_experts}, .f32)) or
        !fc2_global.shape().eql(.init(.{num_experts}, .f32)) or
        !isGlobalOrPerExpertScale(fc1_act_global, num_experts) or
        !isGlobalOrPerExpertScale(fc2_act_global, num_experts))
    {
        return error.InvalidShape;
    }

    return .{
        .num_tokens = tokens,
        .hidden_size = hidden_size,
        .intermediate_size = intermediate_size,
        .num_experts = @intCast(num_experts),
        .top_k = @intCast(top_k),
        .activation = cutlassActivation(options.activation),
        .enable_pdl = options.enable_pdl,
        .fc1_act_per_expert = fc1_act_global.rank() == 1,
        .fc2_act_per_expert = fc2_act_global.rank() == 1,
        .gemm1_tactic = options.gemm1_tactic,
        .gemm2_tactic = options.gemm2_tactic,
    };
}

/// Routed NVFP4 CUTLASS MoE. Weights are logical E2M1 tensors in [expert, output, input] order.
/// Block scales must use FlashInfer nvfp4_block_scale_interleave layout
/// BF16 activations are dynamically quantized to NVFP4 by the fused runner before each expert GEMM.
pub fn fusedExpertsNvfp4(
    hidden_states: zml.Tensor,
    fc1_weights: zml.Tensor,
    fc2_weights: zml.Tensor,
    topk_weights: zml.Tensor,
    topk_ids: zml.Tensor,
    fc1_act_global: zml.Tensor,
    fc1_weight_block: zml.Tensor,
    fc1_global: zml.Tensor,
    fc2_act_global: zml.Tensor,
    fc2_weight_block: zml.Tensor,
    fc2_global: zml.Tensor,
    options: Options,
) zml.Tensor {
    const runners = currentRunners() catch |e| stdx.debug.panic("Failed to get runners: {}", .{e});
    var attributes = validateInputs(
        hidden_states,
        fc1_weights,
        fc2_weights,
        topk_weights,
        topk_ids,
        fc1_act_global,
        fc1_weight_block,
        fc1_global,
        fc2_act_global,
        fc2_weight_block,
        fc2_global,
        options,
    ) catch |e| stdx.debug.panic("Invalid inputs: {}", .{e});

    attributes.runners = @intFromPtr(runners);
    const deviceRunner = runners.ensureRunner(options.workspace_query_device) catch |e| stdx.debug.panic("Failed to ensure runner: {}", .{e});
    const context = makeContext(attributes);
    var requirements = std.mem.zeroes(fi_cutlass_moe.WorkspaceRequirements);
    requirements.struct_size = @sizeOf(fi_cutlass_moe.WorkspaceRequirements);
    checkStatus(
        deviceRunner.api,
        deviceRunner.api.getWorkspaceRequirements(
            deviceRunner.runner,
            &context,
            &requirements,
        ),
    ) catch |e| stdx.debug.panic("Failed to get workspace requirements: {}", .{e});

    const result = routedNvfp4Call.call(
        .{
            .hidden_states = hidden_states,
            .fc1_weights = fc1_weights,
            .fc2_weights = fc2_weights,
            .topk_weights = topk_weights,
            .topk_ids = topk_ids,
            .fc1_act_global = fc1_act_global,
            .fc1_weight_block = fc1_weight_block,
            .fc1_global = fc1_global,
            .fc2_act_global = fc2_act_global,
            .fc2_weight_block = fc2_weight_block,
            .fc2_global = fc2_global,
        },
        .{
            .output = hidden_states.shape(),
            .workspace = .init(.{@as(i64, @intCast(requirements.total_bytes))}, .u8),
        },
        attributes,
    );
    return result.output;
}

pub fn fusedExperts(
    input: zml.Tensor,
    topk_ids: zml.Tensor,
    topk_weights: zml.Tensor,
    gate_up: zml.nn.Linear,
    down: zml.nn.Linear,
    opts: zml.moe.Options,
) zml.Tensor {
    if (comptime !platforms.isEnabled(.cuda)) {
        @panic("FlashInfer CUTLASS MoE is only supported on CUDA platforms");
    }

    if (gate_up.bias != null or down.bias != null) {
        @panic("FlashInfer CUTLASS MoE does not support bias in gate_up or down linear layers");
    }

    const runner_options: Options = .fromGlobalOptions(opts);
    const expert_partition = gate_up.weight.shape().partition(.expert);

    const quant_scheme: ?zml.Quantization.Scheme = if (gate_up.quantization) |q| q.scheme else null;
    stdx.debug.assert(quant_scheme == .nvfp4, "FlashInfer CUTLASS MoE only supports NVFP4, got {?}", .{quant_scheme});

    const gate_up_weight_unpacked = zml.moe.unpackedWeight(gate_up);
    const down_weight_unpacked = zml.moe.unpackedWeight(down);

    // TODO(Corentin): Do error checking on nvfp4
    // Also, maybe pass `zml.nn.Linear` directly
    if (expert_partition.isSharded()) {
        return zml.ops.manualComputation(
            (struct {
                input: zml.Tensor,
                topk_ids: zml.Tensor,
                topk_weights: zml.Tensor,
                gate_up_weight_unpacked: zml.Tensor,
                down_weight_unpacked: zml.Tensor,
                gate_up_input_scale: zml.Tensor,
                gate_up_scales: zml.Tensor,
                gate_up_global_scale: zml.Tensor,
                down_input_scale: zml.Tensor,
                down_scales: zml.Tensor,
                down_global_scale: zml.Tensor,
                activation: zml.moe.Activation,
                enable_pdl: bool,
                gemm1_tactic: i32,
                gemm2_tactic: i32,
                workspace_query_device: i32,

                fn body(
                    self: @This(),
                    _: zml.Shape,
                ) zml.Tensor {
                    const local_num_experts = self.gate_up_weight_unpacked.dim(.expert);
                    const partition_id = zml.ops.partitionId().convert(.i32);
                    const expert_start = partition_id.scale(local_num_experts).convert(.i32);
                    const expert_end = expert_start.addConstant(local_num_experts);

                    const local_route_mask = self.topk_ids
                        .cmp(.GE, expert_start)
                        .logical(.AND, self.topk_ids.cmp(.LT, expert_end));
                    const local_topk_ids = local_route_mask.select(
                        self.topk_ids.sub(expert_start),
                        zml.Tensor.scalar(-1, .i32),
                    );
                    const local_topk_weights = local_route_mask.select(
                        self.topk_weights,
                        zml.Tensor.scalar(-1, self.topk_weights.dtype()),
                    );

                    const local_output = fusedExpertsNvfp4(
                        self.input,
                        self.gate_up_weight_unpacked,
                        self.down_weight_unpacked,
                        local_topk_weights,
                        local_topk_ids,
                        self.gate_up_input_scale,
                        self.gate_up_scales,
                        self.gate_up_global_scale,
                        self.down_input_scale,
                        self.down_scales,
                        self.down_global_scale,
                        .{
                            .workspace_query_device = self.workspace_query_device,
                            .activation = self.activation,
                            .enable_pdl = self.enable_pdl,
                            .gemm1_tactic = self.gemm1_tactic,
                            .gemm2_tactic = self.gemm2_tactic,
                        },
                    );
                    const local_reshaped = local_output
                        .reshape(self.input.shape().dims())
                        .withTags(.{ .t, .d });
                    return zml.ops.allReduce(local_reshaped, zml.Tensor.add);
                }
            }).body,
            .{
                .input = input,
                .topk_ids = topk_ids,
                .topk_weights = topk_weights,
                .gate_up_weight_unpacked = gate_up_weight_unpacked,
                .down_weight_unpacked = down_weight_unpacked,
                .gate_up_input_scale = gate_up.quantization.?.input_scale.?.asMultiplier(),
                .gate_up_scales = gate_up.quantization.?.scales,
                .gate_up_global_scale = gate_up.quantization.?.global_scale.?.asMultiplier(),
                .down_input_scale = down.quantization.?.input_scale.?.asMultiplier(),
                .down_scales = down.quantization.?.scales,
                .down_global_scale = down.quantization.?.global_scale.?.asMultiplier(),
                .activation = runner_options.activation,
                .enable_pdl = runner_options.enable_pdl,
                .gemm1_tactic = runner_options.gemm1_tactic,
                .gemm2_tactic = runner_options.gemm2_tactic,
                .workspace_query_device = runner_options.workspace_query_device,
            },
            input.shape(),
        );
    }

    return fusedExpertsNvfp4(
        input,
        gate_up_weight_unpacked,
        down_weight_unpacked,
        topk_weights,
        topk_ids,
        gate_up.quantization.?.input_scale.?.asMultiplier(),
        gate_up.quantization.?.scales,
        gate_up.quantization.?.global_scale.?.asMultiplier(),
        down.quantization.?.input_scale.?.asMultiplier(),
        down.quantization.?.scales,
        down.quantization.?.global_scale.?.asMultiplier(),
        runner_options,
    );
}
