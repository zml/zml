const std = @import("std");

const fi_cutlass_moe = @import("platforms/cuda/flashinfer_cutlass_moe");
const platforms = @import("platforms");
const zml = @import("../zml.zig");

const log = std.log.scoped(.moe_cutlass_flashinfer);

pub const autoTactic: i32 = -1;

pub const Options = struct {
    /// Device used at graph-construction time to select the architecture
    /// library and size the XLA-owned scratch buffer.
    workspace_query_device: i32 = 0,
    activation: Activation = .swiglu,
    enable_pdl: bool = false,
    gemm1_tactic: i32 = autoTactic,
    /// GEMM2 uses FlashInfer's absolute tactic index. Query tacticCounts() to
    /// obtain the first valid GEMM2 index.
    gemm2_tactic: i32 = autoTactic,
};

pub const Activation = enum {
    swiglu,
    geglu,
    geglu_tanh,
    swiglu_step,
    relu2,
};

pub const Parameters = struct {
    num_experts_per_tok: u32,
    activation: ActivationMode,
    workspace_query_device: i32 = 0,
    enable_pdl: bool = false,
    gemm1_tactic: i32 = autoTactic,
    gemm2_tactic: i32 = autoTactic,

    pub const ActivationMode = enum {
        silu,
        relu,
        gelu,
    };

    pub const InitOptions = struct {
        num_experts_per_tok: u32,
        activation: ActivationMode,
    };

    pub fn init(opts: InitOptions) Parameters {
        return .{
            .num_experts_per_tok = opts.num_experts_per_tok,
            .activation = opts.activation,
        };
    }

    pub fn runnerOptions(self: Parameters) !Options {
        return .{
            .workspace_query_device = self.workspace_query_device,
            .activation = switch (self.activation) {
                .silu => .swiglu,
                // Tensor.gelu() is ZML's tanh GELU approximation.
                .gelu => .geglu_tanh,
                .relu => .relu2,
            },
            .enable_pdl = self.enable_pdl,
            .gemm1_tactic = self.gemm1_tactic,
            .gemm2_tactic = self.gemm2_tactic,
        };
    }
};

pub const Metadata = struct {
    pub const InitOptions = struct {};

    pub fn init(_: InitOptions) Metadata {
        return .{};
    }

    pub fn initBuffer(
        _: Metadata,
        _: std.Io,
        _: *const zml.Platform,
    ) !zml.Bufferized(Metadata) {
        return {};
    }
};

pub fn deinitBuffer(_: *zml.Bufferized(Metadata)) void {}

pub const Nvfp4Scales = struct {
    /// Dynamic activation quantization multiplier, scalar or per expert.
    fc1_act_global: zml.Tensor,
    /// Interleaved E4M3 scales with shape returned by fc1BlockScaleShape().
    fc1_weight_block: zml.Tensor,
    /// Final GEMM1 output alpha, one value per expert.
    fc1_global: zml.Tensor,
    /// Dynamic activation quantization multiplier, scalar or per expert.
    fc2_act_global: zml.Tensor,
    /// Interleaved E4M3 scales with shape returned by fc2BlockScaleShape().
    fc2_weight_block: zml.Tensor,
    /// Final GEMM2 output alpha, one value per expert.
    fc2_global: zml.Tensor,
};

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

const Bf16Input = struct {
    hidden_states: zml.Tensor,
    fc1_weights: zml.Tensor,
    fc2_weights: zml.Tensor,
    topk_weights: zml.Tensor,
    topk_ids: zml.Tensor,
};

const Output = struct {
    output: zml.Shape,
    workspace: zml.Shape,
};

const Attributes = struct {
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

pub const Variant = enum {
    bf16xbf16,
    nvfp4xnvfp4,
};

const maxCudaDevices = 16;
var loaded = false;
var runners: [std.meta.fields(Variant).len][maxCudaDevices]?DeviceRunner =
    @splat(@splat(null));

fn checkStatus(api: *const fi_cutlass_moe.Api, status: fi_cutlass_moe.Status) !void {
    if (status == fi_cutlass_moe.C.ZML_FI_CUTLASS_MOE_STATUS_SUCCESS) return;
    if (api.lastError()) |message| {
        log.err("FlashInfer CUTLASS MoE failed (status {d}): {s}", .{
            status,
            std.mem.span(message),
        });
    } else {
        log.err("FlashInfer CUTLASS MoE failed with status {d}", .{status});
    }
    return switch (status) {
        fi_cutlass_moe.C.ZML_FI_CUTLASS_MOE_STATUS_INVALID_ARGUMENT => error.InvalidArgument,
        fi_cutlass_moe.C.ZML_FI_CUTLASS_MOE_STATUS_UNSUPPORTED => error.UnsupportedArchitecture,
        fi_cutlass_moe.C.ZML_FI_CUTLASS_MOE_STATUS_CUDA_ERROR => error.Cuda,
        else => error.FlashinferCutlassMoe,
    };
}

fn runnerOptions(device: i32, variant: Variant) fi_cutlass_moe.RunnerOptions {
    var options = std.mem.zeroes(fi_cutlass_moe.RunnerOptions);
    options.struct_size = @sizeOf(fi_cutlass_moe.RunnerOptions);
    options.activation_dtype = fi_cutlass_moe.C.ZML_FI_CUTLASS_MOE_DTYPE_BF16;
    options.weight_dtype = switch (variant) {
        .bf16xbf16 => fi_cutlass_moe.C.ZML_FI_CUTLASS_MOE_DTYPE_BF16,
        .nvfp4xnvfp4 => fi_cutlass_moe.C.ZML_FI_CUTLASS_MOE_DTYPE_PACKED_FP4,
    };
    options.output_dtype = fi_cutlass_moe.C.ZML_FI_CUTLASS_MOE_DTYPE_BF16;
    options.device = device;
    options.use_fused_finalize = 1;
    return options;
}

fn ensureRunner(device: i32, variant: Variant) !DeviceRunner {
    if (!loaded) return error.BackendNotLoaded;
    if (device < 0 or device >= maxCudaDevices) return error.UnsupportedDevice;

    const index: usize = @intCast(device);
    const variantIndex: usize = @intFromEnum(variant);
    if (runners[variantIndex][index]) |runner| return runner;

    const api = try fi_cutlass_moe.apiForDevice(device);
    const options = runnerOptions(device, variant);
    var runner: ?*fi_cutlass_moe.Runner = null;
    try checkStatus(api, api.runnerCreate(&options, &runner));
    const result: DeviceRunner = .{
        .api = api,
        .runner = runner orelse return error.RunnerInitializationFailed,
    };
    runners[variantIndex][index] = result;
    return result;
}

fn fi_cutlass_moeActivation(activation: Activation) i32 {
    return switch (activation) {
        .swiglu => fi_cutlass_moe.C.ZML_FI_CUTLASS_MOE_ACTIVATION_SWIGLU,
        .geglu => fi_cutlass_moe.C.ZML_FI_CUTLASS_MOE_ACTIVATION_GEGLU,
        .geglu_tanh => fi_cutlass_moe.C.ZML_FI_CUTLASS_MOE_ACTIVATION_GEGLU_TANH,
        .swiglu_step => fi_cutlass_moe.C.ZML_FI_CUTLASS_MOE_ACTIVATION_SWIGLU_STEP,
        .relu2 => fi_cutlass_moe.C.ZML_FI_CUTLASS_MOE_ACTIVATION_RELU2,
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

fn ffiCall(
    call_frame: *zml.pjrt.ffi.CallFrame,
    input: zml.pjrtx.TensorToCustomCallBuffer(Input),
    output: zml.pjrtx.ShapeToCustomCallBuffer(Output),
    attributes: Attributes,
) !?*zml.pjrt.ffi.Error {
    const device = try call_frame.ctx.getDeviceOrdinal(call_frame.api);
    const deviceRunner = try ensureRunner(device, .nvfp4xnvfp4);
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

const routedNvfp4Call = zml.ops.CustomCall(Input, Output, Attributes, ffiCall, .{
    .name = "flashinfer_cutlass_nvfp4_routed_moe",
    // Expert sharding is owned by moe.forwardMoe's outer manual computation.
    // A second sharding-aware wrapper would bind the same mesh axis twice.
    .sharding_aware = false,
    .has_side_effect = false,
});

fn ffiCallBf16(
    call_frame: *zml.pjrt.ffi.CallFrame,
    input: zml.pjrtx.TensorToCustomCallBuffer(Bf16Input),
    output: zml.pjrtx.ShapeToCustomCallBuffer(Output),
    attributes: Attributes,
) !?*zml.pjrt.ffi.Error {
    const device = try call_frame.ctx.getDeviceOrdinal(call_frame.api);
    const deviceRunner = try ensureRunner(device, .bf16xbf16);
    const context = makeContext(attributes);

    var io = std.mem.zeroes(fi_cutlass_moe.Io);
    io.struct_size = @sizeOf(fi_cutlass_moe.Io);
    io.input = input.hidden_states.ptr;
    io.token_selected_experts = @ptrCast(@alignCast(input.topk_ids.ptr));
    io.token_final_scales = @ptrCast(@alignCast(input.topk_weights.ptr));
    io.fc1_expert_weights = input.fc1_weights.ptr;
    io.fc2_expert_weights = input.fc2_weights.ptr;
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

const routedBf16Call = zml.ops.CustomCall(Bf16Input, Output, Attributes, ffiCallBf16, .{
    .name = "flashinfer_cutlass_bf16_routed_moe",
    .sharding_aware = false,
    .has_side_effect = false,
});

pub fn load(allocator: std.mem.Allocator, io: std.Io) !void {
    if (comptime platforms.isEnabled(.cuda)) {
        try fi_cutlass_moe.load(allocator, io);
        loaded = true;
        return;
    }
    return error.UnsupportedPlatform;
}

pub fn register(platform: *const zml.Platform) !void {
    if (comptime platforms.isEnabled(.cuda)) {
        try routedNvfp4Call.register(platform);
        try routedBf16Call.register(platform);
        return;
    }
    return error.UnsupportedPlatform;
}

pub fn isAvailable(platform: *const zml.Platform) bool {
    if (comptime !platforms.isEnabled(.cuda)) return false;
    if (!loaded or platform.target != .cuda) return false;

    const devices = platform.pjrt_client.devices(platform.pjrt_api);
    if (devices.len == 0) return false;
    const cc = zml.platform.cuda.tryGetComputeCapabilities(platform, devices[0]) orelse
        return false;
    return std.mem.eql(u8, cc, "9.0") or
        std.mem.eql(u8, cc, "10.0") or
        std.mem.eql(u8, cc, "12.0");
}

pub fn isNvfp4Available(platform: *const zml.Platform) bool {
    if (!isAvailable(platform)) return false;

    const devices = platform.pjrt_client.devices(platform.pjrt_api);
    const cc = zml.platform.cuda.tryGetComputeCapabilities(platform, devices[0]) orelse
        return false;
    return std.mem.eql(u8, cc, "10.0") or
        std.mem.eql(u8, cc, "12.0");
}

pub fn tacticCounts(
    device: i32,
    variant: Variant,
) !struct { gemm1: i32, gemm2: i32 } {
    const deviceRunner = try ensureRunner(device, variant);
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
    activation: Activation,
) zml.Shape {
    const rows = switch (activation) {
        .swiglu, .geglu, .geglu_tanh, .swiglu_step => 2 * intermediate_size,
        .relu2 => intermediate_size,
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
    scales: Nvfp4Scales,
    options: Options,
) !Attributes {
    if (hidden_states.dtype() != .bf16 or
        fc1_weights.dtype() != .f4e2m1 or
        fc2_weights.dtype() != .f4e2m1 or
        topk_weights.dtype() != .f32 or
        topk_ids.dtype() != .i32 or
        scales.fc1_weight_block.dtype() != .f8e4m3fn or
        scales.fc2_weight_block.dtype() != .f8e4m3fn or
        scales.fc1_global.dtype() != .f32 or
        scales.fc2_global.dtype() != .f32)
    {
        return error.UnsupportedType;
    }
    if (hidden_states.rank() != 3 or
        fc1_weights.rank() != 3 or
        fc2_weights.rank() != 3 or
        topk_weights.rank() != 3 or
        topk_ids.rank() != 3)
    {
        return error.InvalidShape;
    }

    const batch = hidden_states.dim(0);
    const sequence = hidden_states.dim(1);
    const hidden_size = hidden_states.dim(2);
    const num_experts = fc1_weights.dim(0);
    const fc1_rows = fc1_weights.dim(1);
    const intermediate_size = switch (options.activation) {
        .swiglu, .geglu, .geglu_tanh, .swiglu_step => @divExact(fc1_rows, 2),
        .relu2 => fc1_rows,
    };
    const top_k = topk_ids.dim(2);

    if (batch <= 0 or sequence <= 0 or hidden_size <= 0 or
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
        topk_weights.dim(0) != batch or
        topk_weights.dim(1) != sequence or
        topk_weights.dim(2) != top_k or
        topk_ids.dim(0) != batch or
        topk_ids.dim(1) != sequence or
        !scales.fc1_weight_block.shape().eql(fc1BlockScaleShape(
            num_experts,
            hidden_size,
            intermediate_size,
            options.activation,
        )) or
        !scales.fc2_weight_block.shape().eql(fc2BlockScaleShape(
            num_experts,
            hidden_size,
            intermediate_size,
        )) or
        !scales.fc1_global.shape().eql(.init(.{num_experts}, .f32)) or
        !scales.fc2_global.shape().eql(.init(.{num_experts}, .f32)) or
        !isGlobalOrPerExpertScale(scales.fc1_act_global, num_experts) or
        !isGlobalOrPerExpertScale(scales.fc2_act_global, num_experts))
    {
        return error.InvalidShape;
    }

    return .{
        .num_tokens = batch * sequence,
        .hidden_size = hidden_size,
        .intermediate_size = intermediate_size,
        .num_experts = @intCast(num_experts),
        .top_k = @intCast(top_k),
        .activation = fi_cutlass_moeActivation(options.activation),
        .enable_pdl = options.enable_pdl,
        .fc1_act_per_expert = scales.fc1_act_global.rank() == 1,
        .fc2_act_per_expert = scales.fc2_act_global.rank() == 1,
        .gemm1_tactic = options.gemm1_tactic,
        .gemm2_tactic = options.gemm2_tactic,
    };
}

/// Runs a routed NVFP4 CUTLASS MoE. Weights are logical E2M1 tensors in
/// [expert, output, input] order. Block scales must already use FlashInfer's
/// `nvfp4_block_scale_interleave` layout; this function does not quantize model
/// weights at execution time. BF16 activations are dynamically quantized to
/// NVFP4 by the fused runner before each expert GEMM.
pub fn fusedExpertsImpl(
    hidden_states: zml.Tensor,
    fc1_weights: zml.Tensor,
    fc2_weights: zml.Tensor,
    topk_weights: zml.Tensor,
    topk_ids: zml.Tensor,
    scales: Nvfp4Scales,
    options: Options,
) !zml.Tensor {
    const attributes = try validateInputs(
        hidden_states,
        fc1_weights,
        fc2_weights,
        topk_weights,
        topk_ids,
        scales,
        options,
    );
    const deviceRunner = try ensureRunner(options.workspace_query_device, .nvfp4xnvfp4);
    const context = makeContext(attributes);
    var requirements = std.mem.zeroes(fi_cutlass_moe.WorkspaceRequirements);
    requirements.struct_size = @sizeOf(fi_cutlass_moe.WorkspaceRequirements);
    try checkStatus(
        deviceRunner.api,
        deviceRunner.api.getWorkspaceRequirements(
            deviceRunner.runner,
            &context,
            &requirements,
        ),
    );

    const result = routedNvfp4Call.call(
        .{
            .hidden_states = hidden_states,
            .fc1_weights = fc1_weights,
            .fc2_weights = fc2_weights,
            .topk_weights = topk_weights,
            .topk_ids = topk_ids,
            .fc1_act_global = scales.fc1_act_global,
            .fc1_weight_block = scales.fc1_weight_block,
            .fc1_global = scales.fc1_global,
            .fc2_act_global = scales.fc2_act_global,
            .fc2_weight_block = scales.fc2_weight_block,
            .fc2_global = scales.fc2_global,
        },
        .{
            .output = hidden_states.shape(),
            .workspace = .init(.{@as(i64, @intCast(requirements.total_bytes))}, .u8),
        },
        attributes,
    );
    return result.output;
}

/// Runs a routed BF16 x BF16 CUTLASS MoE with ordinary contiguous row-major
/// [expert, output, input] weights. This is the non-quantized FlashInfer path
/// used on Hopper and Blackwell; unlike TRTLLM Gen it does not require specific layout
pub fn fusedExpertsBf16(
    hidden_states: zml.Tensor,
    fc1_weights: zml.Tensor,
    fc2_weights: zml.Tensor,
    topk_weights: zml.Tensor,
    topk_ids: zml.Tensor,
    options: Options,
) !zml.Tensor {
    if (hidden_states.dtype() != .bf16 or
        fc1_weights.dtype() != .bf16 or
        fc2_weights.dtype() != .bf16 or
        topk_weights.dtype() != .f32 or
        topk_ids.dtype() != .i32 or
        hidden_states.rank() != 3 or
        fc1_weights.rank() != 3 or
        fc2_weights.rank() != 3 or
        topk_weights.rank() != 3 or
        topk_ids.rank() != 3)
    {
        return error.InvalidInput;
    }

    const batch = hidden_states.dim(0);
    const sequence = hidden_states.dim(1);
    const hiddenSize = hidden_states.dim(2);
    const numExperts = fc1_weights.dim(0);
    const fc1Rows = fc1_weights.dim(1);
    const intermediateSize = switch (options.activation) {
        .swiglu, .geglu, .geglu_tanh, .swiglu_step => @divExact(fc1Rows, 2),
        .relu2 => fc1Rows,
    };
    const topK = topk_ids.dim(2);
    if (batch <= 0 or sequence <= 0 or hiddenSize <= 0 or
        numExperts <= 0 or intermediateSize <= 0 or
        topK <= 0 or topK > numExperts or
        fc1_weights.dim(2) != hiddenSize or
        fc2_weights.dim(0) != numExperts or
        fc2_weights.dim(1) != hiddenSize or
        fc2_weights.dim(2) != intermediateSize or
        topk_weights.dim(0) != batch or
        topk_weights.dim(1) != sequence or
        topk_weights.dim(2) != topK or
        topk_ids.dim(0) != batch or
        topk_ids.dim(1) != sequence)
    {
        return error.InvalidShape;
    }

    const attributes: Attributes = .{
        .num_tokens = batch * sequence,
        .hidden_size = hiddenSize,
        .intermediate_size = intermediateSize,
        .num_experts = @intCast(numExperts),
        .top_k = @intCast(topK),
        .activation = fi_cutlass_moeActivation(options.activation),
        .enable_pdl = options.enable_pdl,
        .fc1_act_per_expert = false,
        .fc2_act_per_expert = false,
        .gemm1_tactic = options.gemm1_tactic,
        .gemm2_tactic = options.gemm2_tactic,
    };
    const deviceRunner = try ensureRunner(options.workspace_query_device, .bf16xbf16);
    const context = makeContext(attributes);
    var requirements = std.mem.zeroes(fi_cutlass_moe.WorkspaceRequirements);
    requirements.struct_size = @sizeOf(fi_cutlass_moe.WorkspaceRequirements);
    try checkStatus(
        deviceRunner.api,
        deviceRunner.api.getWorkspaceRequirements(
            deviceRunner.runner,
            &context,
            &requirements,
        ),
    );

    const result = routedBf16Call.call(
        .{
            .hidden_states = hidden_states,
            .fc1_weights = fc1_weights,
            .fc2_weights = fc2_weights,
            .topk_weights = topk_weights,
            .topk_ids = topk_ids,
        },
        .{
            .output = hidden_states.shape(),
            .workspace = .init(.{@as(i64, @intCast(requirements.total_bytes))}, .u8),
        },
        attributes,
    );
    return result.output;
}
