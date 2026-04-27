//! ZML integration for the Triton and Mosaic-TPU kernel DSLs.
//!
//! Each kernel declares its `inputs` and `outputs` by name in its spec.
//! The factory generates `Inputs` (struct of `Tensor`), `Outputs` (struct
//! of `Shape`), and `Results` (struct of `Tensor`) types from those names,
//! so every callsite is by-name and Zig catches missing/typo'd/extra
//! arguments at compile time.

const std = @import("std");

const mlir = @import("mlir");
const dialects = @import("mlir/dialects");

const triton_builder = @import("kernels/triton/builder");
const mosaic_tpu_builder = @import("kernels/mosaic_tpu/builder");
const tpu_dialect = @import("mlir/dialects/mosaic_tpu");

const DataType = @import("dtype.zig").DataType;
const Shape = @import("shape.zig").Shape;
const ops = @import("ops.zig");
const Tensor = @import("tensor.zig").Tensor;
const CompilationContext = @import("module.zig").CompilationContext;
const mlirx = @import("mlirx.zig");

/// Build a struct type with one field per name, all of type `FieldT`.
fn StructOf(comptime names: []const [:0]const u8, comptime FieldT: type) type {
    var name_slices: [names.len][]const u8 = undefined;
    for (names, 0..) |n, i| name_slices[i] = n;
    return @Struct(.auto, null, &name_slices, &@splat(FieldT), &@splat(.{}));
}

pub const triton = struct {
    pub const Builder = triton_builder.Builder;
    pub const Value = triton_builder.Value;
    pub const DType = triton_builder.DType;
    pub const ArgSpec = triton_builder.ArgSpec;
    pub const FinishError = triton_builder.FinishError;

    pub fn from(dt: DataType) DType {
        return switch (dt) {
            .bool => .i1,
            .i8, .u8 => .i8,
            .i16, .u16 => .i16,
            .i32, .u32 => .i32,
            .i64, .u64 => .i64,
            .f16 => .f16,
            .bf16 => .bf16,
            .f32 => .f32,
            .f64 => .f64,
            .f8e4m3fn => .f8e4m3fn,
            .f8e5m2 => .f8e5m2,
            else => std.debug.panic("zml.kernel.triton.from: dtype {s} has no Triton equivalent", .{@tagName(dt)}),
        };
    }

    pub fn Spec(comptime Config: type) type {
        return struct {
            name: [:0]const u8,
            inputs: []const [:0]const u8,
            outputs: []const [:0]const u8,
            run: *const fn (*Builder, Config) FinishError!void,
        };
    }

    pub fn Kernel(
        comptime ConfigT: type,
        comptime spec: Spec(ConfigT),
    ) type {
        return struct {
            pub const name: [:0]const u8 = spec.name;
            pub const Config = ConfigT;
            pub const Inputs = StructOf(spec.inputs, Tensor);
            pub const Outputs = StructOf(spec.outputs, Shape);
            pub const Results = StructOf(spec.outputs, Tensor);

            pub const CallOpts = struct {
                cfg: ConfigT,
                grid: [3]i32,
                num_stages: i32,
                num_warps: i32,
                output_operand_aliases: @FieldType(ops.TritonOps, "output_operand_aliases") = &.{},
                debug: bool = false,
            };

            pub fn emit(
                allocator: std.mem.Allocator,
                ctx: *mlir.Context,
                cfg: ConfigT,
            ) ![:0]const u8 {
                var b = try triton_builder.Builder.open(allocator, ctx, name);
                defer b.deinit();
                try spec.run(&b, cfg);
                return b.finish(&.{});
            }

            pub fn call(inputs: Inputs, outputs: Outputs, opts: CallOpts) Results {
                const cur = CompilationContext.current();

                const ttir = emit(cur.allocator, cur.mlir_ctx, opts.cfg) catch |err|
                    std.debug.panic("zml.kernel.triton.Kernel({s}).call: emit failed: {}", .{ name, err });
                defer cur.allocator.free(ttir);

                var inputs_arr: [spec.inputs.len]Tensor = undefined;
                inline for (spec.inputs, 0..) |fname, i| inputs_arr[i] = @field(inputs, fname);

                var outputs_arr: [spec.outputs.len]Shape = undefined;
                inline for (spec.outputs, 0..) |fname, i| outputs_arr[i] = @field(outputs, fname);

                const tensor_results = ops.triton(inputs_arr, outputs_arr, .{
                    .debug = opts.debug,
                    .name = name,
                    .ir = ttir,
                    .grid = opts.grid,
                    .num_stages = opts.num_stages,
                    .num_warps = opts.num_warps,
                    .output_operand_aliases = opts.output_operand_aliases,
                });

                var results: Results = undefined;
                inline for (spec.outputs, 0..) |fname, i| @field(results, fname) = tensor_results[i];
                return results;
            }
        };
    }
};

pub const mosaic_tpu = struct {
    pub const Builder = mosaic_tpu_builder.Builder;
    pub const Value = mosaic_tpu_builder.Value;
    pub const DType = mosaic_tpu_builder.DType;
    pub const ArgSpec = mosaic_tpu_builder.ArgSpec;
    pub const FinishError = mosaic_tpu_builder.FinishError;

    pub const MemorySpace = mosaic_tpu_builder.MemorySpace;
    pub const ReductionKind = mosaic_tpu_builder.ReductionKind;
    pub const ContractPrecision = mosaic_tpu_builder.ContractPrecision;
    pub const RoundingMode = mosaic_tpu_builder.RoundingMode;
    pub const DimensionSemantics = mosaic_tpu_builder.DimensionSemantics;
    pub const CoreType = mosaic_tpu_builder.CoreType;
    pub const PipelineMode = mosaic_tpu_builder.PipelineMode;
    pub const RevisitMode = mosaic_tpu_builder.RevisitMode;
    pub const CombiningKind = mosaic_tpu_builder.CombiningKind;

    pub fn from(dt: DataType) DType {
        return switch (dt) {
            .bool => .i1,
            .i8, .u8 => .i8,
            .i16, .u16 => .i16,
            .i32, .u32 => .i32,
            .i64, .u64 => .i64,
            .f16 => .f16,
            .bf16 => .bf16,
            .f32 => .f32,
            .f64 => .f64,
            .f8e4m3fn => .f8e4m3fn,
            .f8e5m2 => .f8e5m2,
            else => std.debug.panic("zml.kernel.mosaic_tpu.from: dtype {s} has no Mosaic-TPU equivalent", .{@tagName(dt)}),
        };
    }

    pub fn Spec(comptime Config: type) type {
        return struct {
            name: [:0]const u8,
            inputs: []const [:0]const u8,
            outputs: []const [:0]const u8,
            run: *const fn (*Builder, Config) FinishError!void,
        };
    }

    pub fn Kernel(
        comptime ConfigT: type,
        comptime spec: Spec(ConfigT),
    ) type {
        return struct {
            pub const name: [:0]const u8 = spec.name;
            pub const Config = ConfigT;
            pub const Inputs = StructOf(spec.inputs, Tensor);
            pub const Outputs = StructOf(spec.outputs, Shape);
            pub const Results = StructOf(spec.outputs, Tensor);

            pub const CallOpts = struct {
                cfg: ConfigT,
                extras: CallExtras = .{},
            };

            pub fn emit(
                allocator: std.mem.Allocator,
                ctx: *mlir.Context,
                cfg: ConfigT,
            ) ![:0]const u8 {
                var b = try mosaic_tpu_builder.Builder.open(allocator, ctx, name);
                defer b.deinit();
                try spec.run(&b, cfg);
                return b.finishOpts(&.{}, .{ .canonicalize = true });
            }

            pub fn call(inputs: Inputs, outputs: Outputs, opts: CallOpts) Results {
                const cur = CompilationContext.current();

                const ir = emit(cur.allocator, cur.mlir_ctx, opts.cfg) catch |err|
                    std.debug.panic("zml.kernel.mosaic_tpu.Kernel({s}).call: emit failed: {}", .{ name, err });
                defer cur.allocator.free(ir);

                const backend_config = buildBackendConfig(cur.allocator, cur.mlir_ctx, ir, opts.extras) catch |err|
                    std.debug.panic("zml.kernel.mosaic_tpu.Kernel({s}).call: backend_config build failed: {}", .{ name, err });
                defer cur.allocator.free(backend_config);

                var values: [spec.inputs.len]*const mlir.Value = undefined;
                inline for (spec.inputs, 0..) |fname, i| values[i] = @field(inputs, fname).value();

                var res_types: [spec.outputs.len]*const mlir.Type = undefined;
                inline for (spec.outputs, 0..) |fname, i| {
                    const out = @field(outputs, fname);
                    res_types[i] = mlir.rankedTensorType(out.dims(), mlirx.Type.fromDType(cur.mlir_ctx, out.dtype()));
                }

                const op = callTpuCustomCall(&values, &res_types, backend_config, opts.extras);

                var results: Results = undefined;
                inline for (spec.outputs, 0..) |fname, i| {
                    const out = @field(outputs, fname);
                    @field(results, fname) = Tensor._result(out, op.result(i));
                }
                return results;
            }
        };
    }

    pub const CallExtras = struct {
        vmem_limit_bytes: ?i64 = null,
        disable_bounds_checks: ?bool = null,
        disable_semaphore_checks: ?bool = null,
        has_communication: ?bool = null,
        additional_attributes: []const mlir.NamedAttribute = &.{},
        output_operand_aliases: []const dialects.stablehlo.CustomCallOpts.OutputOperandAlias = &.{},
    };

    const CustomCallConfig = struct {
        body: []const u8,
        has_communication: ?bool = null,
        serialization_format: i64 = 1,
        needs_layout_passes: bool = true,
        shape_invariant_numerics: bool = false,
        disable_bounds_checks: ?bool = null,
        disable_semaphore_checks: ?bool = null,
    };

    const ScopedMemoryConfig = struct {
        memory_space: i64 = 1,
        offset: i64 = 0,
        size: i64,
    };

    const BackendConfig = struct {
        custom_call_config: CustomCallConfig,
        scoped_memory_configs: ?[]const ScopedMemoryConfig = null,
    };

    pub fn buildBackendConfig(
        allocator: std.mem.Allocator,
        ctx: *mlir.Context,
        ir: [:0]const u8,
        extras: CallExtras,
    ) ![]u8 {
        const module = try mlir.Module.parse(ctx, ir);
        defer module.deinit();

        tpu_dialect.registerMosaicSerdePass();
        const prev_allow_unregistered = ctx.allowUnregisteredDialects();
        ctx.setAllowUnregisteredDialects(true);
        defer ctx.setAllowUnregisteredDialects(prev_allow_unregistered);

        const pm = mlir.PassManager.init(ctx);
        defer pm.deinit();
        try pm.asOpPassManager().addPipeline("mosaic-serde{serialize=true}");
        try pm.runOnOp(module.operation());

        const Encoder = std.base64.standard.Encoder;
        const b64 = b64: {
            var bytecode: std.Io.Writer.Allocating = .init(allocator);
            defer bytecode.deinit();
            try module.operation().writeBytecode(.{ .desired_emit_version = 0 }, &bytecode.writer);
            const buf = try allocator.alloc(u8, Encoder.calcSize(bytecode.written().len));
            break :b64 Encoder.encode(buf, bytecode.written());
        };
        defer allocator.free(b64);

        const config = BackendConfig{
            .custom_call_config = .{
                .body = b64,
                .has_communication = extras.has_communication,
                .disable_bounds_checks = extras.disable_bounds_checks,
                .disable_semaphore_checks = extras.disable_semaphore_checks,
            },
            .scoped_memory_configs = if (extras.vmem_limit_bytes) |vmem|
                &.{.{ .size = vmem }}
            else
                null,
        };

        var json: std.Io.Writer.Allocating = .init(allocator);
        errdefer json.deinit();

        try json.writer.print("{f}", .{std.json.fmt(config, .{ .emit_null_optional_fields = false })});
        return try json.toOwnedSlice();
    }

    pub fn callTpuCustomCall(
        inputs: []const *const mlir.Value,
        result_types: []const *const mlir.Type,
        backend_config: []const u8,
        extras: CallExtras,
    ) *mlir.Operation {
        const cur = CompilationContext.current();
        return dialects.stablehlo.custom_call(
            cur.mlir_ctx,
            inputs,
            result_types,
            .{
                .call_target_name = "tpu_custom_call",
                .backend_config = .{ .original = backend_config },
                .has_side_effect = false,
                .api_version = .original,
                .additional_attributes = extras.additional_attributes,
                .output_operand_aliases = extras.output_operand_aliases,
            },
            .unknown(cur.mlir_ctx),
        ).appendTo(cur.currentScope().block);
    }
};
