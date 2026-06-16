const std = @import("std");

const stdx = @import("stdx");
const dialects = @import("mlir/dialects");
const mlir = @import("mlir");
const mosaic_tpu_builder = @import("kernels/mosaic_tpu/builder");
const tpu_dialect = @import("mlir/dialects/mosaic_tpu");
const triton_builder = @import("kernels/triton/builder");

const CompilationContext = @import("module.zig").CompilationContext;
const DataType = @import("dtype.zig").DataType;
const mlirx = @import("mlirx.zig");
const ops = @import("ops.zig");
const Shape = @import("shape.zig").Shape;
const Tensor = @import("tensor.zig").Tensor;

/// Build a struct type with one field per name, all of type `FieldT`.
fn StructOf(comptime names: []const [:0]const u8, comptime FieldT: type) type {
    var name_slices: [names.len][]const u8 = undefined;
    for (names, 0..) |n, i| name_slices[i] = n;
    return @Struct(.auto, null, &name_slices, &@splat(FieldT), &@splat(.{}));
}

fn makeKernelContext(comptime dialects_needed: []const []const u8) std.mem.Allocator.Error!*mlir.Context {
    mlir.registerPasses("Transforms");

    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();

    inline for (dialects_needed) |d| {
        mlir.DialectHandle.fromString(d).insertDialect(registry);
    }

    mlir.registerFuncExtensions(registry);

    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();

    return ctx;
}

fn resolveOutputOperandAliases(
    aliases_opt: anytype,
    operand_offset: i64,
) stdx.BoundedArray(dialects.stablehlo.CustomCallOpts.OutputOperandAlias, dialects.stablehlo.CustomCallOpts.MAX_RESULTS) {
    var out: stdx.BoundedArray(dialects.stablehlo.CustomCallOpts.OutputOperandAlias, dialects.stablehlo.CustomCallOpts.MAX_RESULTS) = .empty;

    if (aliases_opt) |a| {
        inline for (@typeInfo(@TypeOf(a)).@"struct".fields, 0..) |field, output_index| {
            if (@field(a, field.name)) |operand| out.appendAssumeCapacity(.{
                .output_index = @intCast(output_index),
                .operand_index = @as(i64, @intCast(@intFromEnum(operand))) + operand_offset,
            });
        }
    }

    return out;
}

pub const triton = struct {
    pub const Builder = triton_builder.Builder;
    pub const Value = triton_builder.Value;
    pub const DType = triton_builder.DType;
    pub const FinishError = triton_builder.FinishError;

    pub fn newContext() std.mem.Allocator.Error!*mlir.Context {
        return makeKernelContext(&triton_builder.dialects_needed);
    }

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

    fn Spec(comptime Config: type) type {
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
                output_operand_aliases: ?ops.CustomCallOutputOperandAliases(Inputs, Outputs) = null,
                debug: bool = false,
            };

            pub fn emit(allocator: std.mem.Allocator, cfg: ConfigT) ![:0]const u8 {
                const ctx = try newContext();
                defer ctx.deinit();

                var b = try triton_builder.Builder.open(allocator, ctx, name);
                defer b.deinit();

                try spec.run(&b, cfg);

                return b.finish(&.{});
            }

            pub fn call(inputs: Inputs, outputs: Outputs, opts: CallOpts) Results {
                const cur = CompilationContext.current();

                const ttir = emit(cur.allocator, opts.cfg) catch |err|
                    std.debug.panic("zml.kernel.triton.Kernel({s}).call: emit failed: {}", .{ name, err });
                defer cur.allocator.free(ttir);

                var inputs_arr: [spec.inputs.len]Tensor = undefined;
                inline for (spec.inputs, 0..) |fname, i| inputs_arr[i] = @field(inputs, fname);

                var outputs_arr: [spec.outputs.len]Shape = undefined;
                inline for (spec.outputs, 0..) |fname, i| outputs_arr[i] = @field(outputs, fname);

                const aliases = resolveOutputOperandAliases(opts.output_operand_aliases, 0);

                const tensor_results = ops.triton(inputs_arr, outputs_arr, .{
                    .debug = opts.debug,
                    .name = name,
                    .ir = ttir,
                    .grid = opts.grid,
                    .num_stages = opts.num_stages,
                    .num_warps = opts.num_warps,
                    .output_operand_aliases = aliases.constSlice(),
                });

                var results: Results = undefined;
                inline for (spec.outputs, 0..) |fname, i| @field(results, fname) = tensor_results[i];
                return results;
            }
        };
    }
};

pub const mosaic_tpu = struct {
    const Builder = mosaic_tpu_builder.Builder;
    const DType = mosaic_tpu_builder.DType;
    const FinishError = mosaic_tpu_builder.FinishError;

    pub fn newContext() std.mem.Allocator.Error!*mlir.Context {
        return makeKernelContext(&mosaic_tpu_builder.dialects_needed);
    }

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

    fn Spec(comptime Config: type) type {
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

            pub const CallExtras = struct {
                vmem_limit_bytes: ?i64 = null,
                disable_bounds_checks: ?bool = null,
                disable_semaphore_checks: ?bool = null,
                has_communication: ?bool = null,
                additional_attributes: []const mlir.NamedAttribute = &.{},
                output_operand_aliases: ?ops.CustomCallOutputOperandAliases(Inputs, Outputs) = null,
                dynamic_grid_bounds: []const Tensor = &.{},
            };

            pub const CallOpts = struct {
                cfg: ConfigT,
                extras: CallExtras = .{},
            };

            pub fn emit(allocator: std.mem.Allocator, cfg: ConfigT) ![:0]const u8 {
                const ctx = try newContext();
                defer ctx.deinit();

                var b = try mosaic_tpu_builder.Builder.open(allocator, ctx, name);
                defer b.deinit();

                try spec.run(&b, cfg);

                return b.finishOpts(&.{}, .{ .canonicalize = true });
            }

            pub fn call(inputs: Inputs, outputs: Outputs, opts: CallOpts) Results {
                const cur = CompilationContext.current();

                const ir = emit(cur.allocator, opts.cfg) catch |err|
                    std.debug.panic("zml.kernel.mosaic_tpu.Kernel({s}).call: emit failed: {}", .{ name, err });
                defer cur.allocator.free(ir);

                const backend_config = buildBackendConfig(cur.allocator, ir, .{
                    .vmem_limit_bytes = opts.extras.vmem_limit_bytes,
                    .disable_bounds_checks = opts.extras.disable_bounds_checks,
                    .disable_semaphore_checks = opts.extras.disable_semaphore_checks,
                    .has_communication = opts.extras.has_communication,
                }) catch |err|
                    std.debug.panic("zml.kernel.mosaic_tpu.Kernel({s}).call: backend_config build failed: {}", .{ name, err });
                defer cur.allocator.free(backend_config);

                var values: [spec.inputs.len]*const mlir.Value = undefined;
                inline for (spec.inputs, 0..) |fname, i| values[i] = @field(inputs, fname).value();

                var res_types: [spec.outputs.len]*const mlir.Type = undefined;
                inline for (&res_types, spec.outputs) |*r, fname| {
                    r.* = mlirx.Type.rankedTensor(cur.mlir_ctx, @field(outputs, fname));
                }

                const aliases = resolveOutputOperandAliases(opts.extras.output_operand_aliases, @intCast(opts.extras.dynamic_grid_bounds.len));

                const op = callTpuCustomCall(.{
                    .inputs = &values,
                    .result_types = &res_types,
                    .backend_config = backend_config,
                    .aliases = aliases.constSlice(),
                    .dynamic_grid_bounds = opts.extras.dynamic_grid_bounds,
                    .additional_attributes = opts.extras.additional_attributes,
                });

                var results: Results = undefined;
                inline for (spec.outputs, 0..) |fname, i| {
                    const out = @field(outputs, fname);
                    @field(results, fname) = Tensor._result(out, op.result(i));
                }
                return results;
            }
        };
    }

    pub const RawCallExtras = struct {
        vmem_limit_bytes: ?i64 = null,
        disable_bounds_checks: ?bool = null,
        disable_semaphore_checks: ?bool = null,
        has_communication: ?bool = null,
        additional_attributes: []const mlir.NamedAttribute = &.{},
        dynamic_grid_bounds: []const Tensor = &.{},
    };

    pub fn callRaw(
        ir: [:0]const u8,
        inputs: []const Tensor,
        outputs: []const Shape,
        extras: RawCallExtras,
    ) []Tensor {
        const cur = CompilationContext.current();
        const allocator = cur.arena.allocator();

        const backend_config = buildBackendConfig(cur.allocator, ir, .{
            .vmem_limit_bytes = extras.vmem_limit_bytes,
            .disable_bounds_checks = extras.disable_bounds_checks,
            .disable_semaphore_checks = extras.disable_semaphore_checks,
            .has_communication = extras.has_communication,
        }) catch |err|
            std.debug.panic("zml.kernel.mosaic_tpu.callRaw: backend_config build failed: {}", .{err});
        defer cur.allocator.free(backend_config);

        const values = allocator.alloc(*const mlir.Value, inputs.len) catch unreachable;
        for (inputs, 0..) |input, i| values[i] = input.value();

        const res_types = allocator.alloc(*const mlir.Type, outputs.len) catch unreachable;
        for (outputs, 0..) |output, i| {
            res_types[i] = mlirx.Type.rankedTensor(cur.mlir_ctx, output);
        }

        const op = callTpuCustomCall(.{
            .inputs = values,
            .result_types = res_types,
            .backend_config = backend_config,
            .aliases = &.{},
            .dynamic_grid_bounds = extras.dynamic_grid_bounds,
            .additional_attributes = extras.additional_attributes,
        });

        const results = allocator.alloc(Tensor, outputs.len) catch unreachable;
        for (outputs, 0..) |output, i| {
            results[i] = Tensor._result(output, op.result(i));
        }
        return results;
    }

    const CustomCallConfig = struct {
        body: []const u8,
        has_communication: ?bool = null,
        collective_id: ?[]const u8 = null,
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
        const BarrierConfig = struct {
            barrier_type: []const u8 = "CUSTOM",
            id: []const u8 = "0",
        };

        barrier_config: ?BarrierConfig = null,
        custom_call_config: CustomCallConfig,
        scoped_memory_configs: ?[]const ScopedMemoryConfig = null,
    };

    const BackendConfigOpts = struct {
        vmem_limit_bytes: ?i64 = null,
        disable_bounds_checks: ?bool = null,
        disable_semaphore_checks: ?bool = null,
        has_communication: ?bool = null,
    };

    fn buildBackendConfig(
        allocator: std.mem.Allocator,
        ir: [:0]const u8,
        opts: BackendConfigOpts,
    ) ![]u8 {
        const ctx = try newContext();
        defer ctx.deinit();

        const module = try mlir.Module.parse(ctx, ir);
        defer module.deinit();
        const communication = tpu_dialect.analyzePotentialCommunication(module.operation());

        tpu_dialect.registerMosaicSerdePass();
        const prev_allow_unregistered = ctx.allowUnregisteredDialects();
        ctx.setAllowUnregisteredDialects(true);
        defer ctx.setAllowUnregisteredDialects(prev_allow_unregistered);

        const pm = mlir.PassManager.init(ctx);
        defer pm.deinit();
        const opm = pm.asOpPassManager();
        inline for (.{ "canonicalize", "cse", "canonicalize", "mosaic-serde{serialize=true}" }) |pass| {
            try opm.addPipeline(pass);
        }
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

        const has_communication = opts.has_communication orelse if (communication.has_communication) true else null;
        const config = BackendConfig{
            .barrier_config = if (communication.has_custom_barrier) .{} else null,
            .custom_call_config = .{
                .body = b64,
                .has_communication = has_communication,
                .collective_id = if (has_communication orelse false) "0" else null,
                .disable_bounds_checks = opts.disable_bounds_checks,
                .disable_semaphore_checks = opts.disable_semaphore_checks,
            },
            .scoped_memory_configs = if (opts.vmem_limit_bytes) |vmem|
                &.{.{ .size = vmem }}
            else
                null,
        };

        var json: std.Io.Writer.Allocating = .init(allocator);
        errdefer json.deinit();

        try json.writer.print("{f}", .{std.json.fmt(config, .{ .emit_null_optional_fields = false })});
        return try json.toOwnedSlice();
    }

    const TpuCustomCallArgs = struct {
        inputs: []const *const mlir.Value,
        result_types: []const *const mlir.Type,
        backend_config: []const u8,
        aliases: []const dialects.stablehlo.CustomCallOpts.OutputOperandAlias,
        dynamic_grid_bounds: []const Tensor,
        additional_attributes: []const mlir.NamedAttribute,
    };

    fn callTpuCustomCall(args: TpuCustomCallArgs) *mlir.Operation {
        const cur = CompilationContext.current();

        var all_inputs: stdx.BoundedArray(*const mlir.Value, dialects.stablehlo.CustomCallOpts.MAX_OPERANDS) = .empty;
        for (args.dynamic_grid_bounds) |t| all_inputs.appendAssumeCapacity(t.value());
        all_inputs.appendSliceAssumeCapacity(args.inputs);

        return dialects.stablehlo.custom_call(
            cur.mlir_ctx,
            all_inputs.constSlice(),
            args.result_types,
            .{
                .call_target_name = "tpu_custom_call",
                .backend_config = .{ .original = args.backend_config },
                .has_side_effect = false,
                .api_version = .original,
                .additional_attributes = args.additional_attributes,
                .output_operand_aliases = args.aliases,
            },
            .unknown(cur.mlir_ctx),
        ).appendTo(cur.currentScope().block);
    }
};
