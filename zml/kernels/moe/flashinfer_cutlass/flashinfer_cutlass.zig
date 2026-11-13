const std = @import("std");

const fi_cutlass_moe = @import("platforms/cuda/flashinfer_cutlass_moe");
const platforms = @import("platforms");
const stdx = @import("stdx");

const nn = @import("zml/nn");
const Quantization = @import("zml/quantization").Quantization;

const zml = @import("zml/core");
const GlobalOptions = @import("zml/kernels/moe/api").Options;
const Activation = @import("zml/kernels/moe/api").Activation;
const LocalRouting = @import("zml/kernels/moe/api").LocalRouting;
const unpackedWeight = @import("zml/kernels/moe/weights").unpackedWeight;
const contract = @import("zml/core").flashinfer_cutlass.contract;
const Attributes = contract.Attributes;
const runtime = @import("zml/core").flashinfer_cutlass.runtime;

const log = std.log.scoped(.@"zml/moe/flashinfer_cutlass");

pub const auto_tactic: i32 = -1;

pub const Options = struct {
    /// Device used at graph-construction time to select the architecture
    /// library and size the XLA-owned scratch buffer.
    workspace_query_device: i32,
    activation: Activation,
    enable_pdl: bool,
    gemm1_tactic: i32,
    /// GEMM2 uses FlashInfer's absolute tactic index. Query tacticCounts() to
    /// obtain the first valid GEMM2 index.
    gemm2_tactic: i32,

    pub fn fromGlobalOptions(opts: GlobalOptions) Options {
        return .{
            .workspace_query_device = 0,
            .activation = opts.activation,
            .enable_pdl = false,
            .gemm1_tactic = auto_tactic,
            .gemm2_tactic = auto_tactic,
        };
    }
};

pub fn validateOptions(opts: GlobalOptions) void {
    stdx.debug.assert(!opts.quantize_input, "Optional FP8 input quantization requires the Triton MoE backend", .{});
    stdx.debug.assert(opts.routing_weight_placement == .after_down, "Non-Triton MoE backends require routing weights after the down projection", .{});
    stdx.debug.assert(opts.activation == .swiglu, "flashinfer_cutlass backend only accepts swiglu activation, got {}", .{opts.activation});
    stdx.debug.assert(opts.activation.swiglu.limit == null, "Activation thresholds require the Triton MoE backend", .{});
    stdx.debug.assert(opts.activation.swiglu.bias == null, "flashinfer_cutlass backend requires swiglu bias to be null", .{});
    stdx.debug.assert(opts.activation.swiglu.scale == null, "flashinfer_cutlass backend requires swiglu scale to be null", .{});
}

const Inputs = contract.Inputs(zml.Tensor);

const Outputs = contract.Outputs(zml.Shape);

fn currentRunners() !*runtime.Runners {
    const platform = zml.Compiler.current().platform;
    const runners = platform.state.cuda.flashinfer_cutlass_moe_runners;
    return runners orelse error.RunnersNotLoaded;
}

fn cutlassActivation(activation: Activation) i32 {
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

pub fn call(input_tensors: Inputs, output_shapes: Outputs, attributes: Attributes) zml.ops.ShapeToTensor(Outputs) {
    const opts: zml.ops.CustomCallOptions = .{
        .has_side_effect = false,
        .output_operand_aliases = comptime zml.ops.customCallOutputOperandAliases(Inputs, Outputs, null),
        .compute_on_host = contract.compute_on_host,
    };

    return zml.ops.typedCustomCall(
        contract.name,
        opts,
        input_tensors,
        output_shapes,
        attributes,
    );
}

fn computeCapability(platform: *const zml.Platform) !u16 {
    const cc = zml.platform.cuda.computeCapability(platform) orelse return error.UnsupportedPlatform;
    return switch (cc.sm()) {
        100, 103, 120 => |sm| sm,
        else => error.UnsupportedArchitecture,
    };
}

pub fn isAvailable(platform: *const zml.Platform) bool {
    if (platform.target != .cuda) return false;
    if (platform.state.cuda.flashinfer_cutlass_moe_runners == null) return false;
    _ = computeCapability(platform) catch return false;
    return true;
}

pub fn isNvfp4Supported(platform: *const zml.Platform) bool {
    if (platform.state.cuda.flashinfer_cutlass_moe_runners == null) return false;
    const compute_capability = computeCapability(platform) catch return false;
    return switch (compute_capability) {
        100, 103, 120 => true,
        else => false,
    };
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
    const context = runtime.makeContext(attributes);
    var requirements = std.mem.zeroes(fi_cutlass_moe.WorkspaceRequirements);
    requirements.struct_size = @sizeOf(fi_cutlass_moe.WorkspaceRequirements);
    runtime.checkStatus(
        deviceRunner.api,
        deviceRunner.api.getWorkspaceRequirements(
            deviceRunner.runner,
            &context,
            &requirements,
        ),
    ) catch |e| stdx.debug.panic("Failed to get workspace requirements: {}", .{e});

    const result = call(
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
    routing: LocalRouting,
    gate_up: nn.Linear,
    down: nn.Linear,
    opts: GlobalOptions,
) zml.Tensor {
    if (comptime !platforms.isEnabled(.cuda)) {
        @panic("FlashInfer CUTLASS MoE is only supported on CUDA platforms");
    }

    if (gate_up.bias != null or down.bias != null) {
        @panic("FlashInfer CUTLASS MoE does not support bias in gate_up or down linear layers");
    }

    const topk_ids = routing.localIds();
    // The runner ignores routes whose local expert id is -1.
    const topk_weights = topk_ids.cmp(.GE, .scalar(0, .i32)).select(routing.weights, .scalar(-1, routing.weights.dtype()));

    const runner_options: Options = .fromGlobalOptions(opts);

    const quant_scheme: ?Quantization.Scheme = if (gate_up.quantization) |q| q.scheme else null;
    stdx.debug.assert(quant_scheme == .nvfp4, "FlashInfer CUTLASS MoE only supports NVFP4, got {?}", .{quant_scheme});

    const gate_up_weight_unpacked = unpackedWeight(gate_up);
    const down_weight_unpacked = unpackedWeight(down);

    // TODO(Corentin): Do error checking on nvfp4
    // Also, maybe pass `nn.Linear` directly
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
