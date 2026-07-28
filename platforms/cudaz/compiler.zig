const std = @import("std");

const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const dialects = @import("mlir/dialects");
const mlir = @import("mlir");
const runfiles = @import("runfiles");

const log = std.log.scoped(.@"zml/platforms/cudaz/compiler");
const zig_runtime_config = "zml/platforms/cudaz/zig_runtime.txt";

pub const Error = error{
    RunfilesUnavailable,
    ToolchainConfigUnavailable,
    ZigExecutableUnavailable,
    ZigLibUnavailable,
    KernelSourceUnavailable,
    KernelLibraryUnavailable,
    ZigCompilationFailed,
    InvalidKernelIr,
    InvalidMlirProgram,
    MissingMainFunction,
    UnsupportedGraph,
    UnsupportedOutputType,
};

pub const OutputSpec = struct {
    element_type: c.PJRT_Buffer_Type,
    dims: []i64,

    fn clone(self: OutputSpec, allocator: std.mem.Allocator) !OutputSpec {
        return .{
            .element_type = self.element_type,
            .dims = try allocator.dupe(i64, self.dims),
        };
    }

    fn deinit(self: OutputSpec, allocator: std.mem.Allocator) void {
        allocator.free(self.dims);
    }
};

pub const GeneratedKernel = struct {
    source: []u8,
    output_specs: []OutputSpec,
    scratch_size: usize,
    input_count: usize,

    pub fn deinit(self: *GeneratedKernel, allocator: std.mem.Allocator) void {
        allocator.free(self.source);
        deinitOutputSpecs(allocator, self.output_specs);
        self.* = undefined;
    }
};

const KernelType = enum {
    u8,
    i32,
    u32,
    f32,

    fn zigName(self: KernelType) []const u8 {
        return @tagName(self);
    }

    fn byteSize(self: KernelType) usize {
        return switch (self) {
            .u8 => @sizeOf(u8),
            .i32 => @sizeOf(i32),
            .u32 => @sizeOf(u32),
            .f32 => @sizeOf(f32),
        };
    }

    fn alignment(self: KernelType) usize {
        return switch (self) {
            .u8 => @alignOf(u8),
            .i32 => @alignOf(i32),
            .u32 => @alignOf(u32),
            .f32 => @alignOf(f32),
        };
    }
};

const ExternalBuffer = struct {
    index: usize,
    element_type: KernelType,
    elements: usize,
};

const ScratchBuffer = struct {
    offset: usize,
    element_type: KernelType,
    elements: usize,
};

const BufferRef = union(enum) {
    argument: ExternalBuffer,
    output: ExternalBuffer,
    scratch: ScratchBuffer,

    fn elementType(self: BufferRef) KernelType {
        return switch (self) {
            inline else => |buffer| buffer.element_type,
        };
    }

    fn elements(self: BufferRef) usize {
        return switch (self) {
            inline else => |buffer| buffer.elements,
        };
    }
};

const Matmul = struct {
    lhs: BufferRef,
    rhs: BufferRef,
    output: BufferRef,
    m: usize,
    n: usize,
    k: usize,
};

const AddBiasRelu = struct {
    values: BufferRef,
    bias: BufferRef,
    rows: usize,
    columns: usize,
};

const ArgMax = struct {
    values: BufferRef,
    output: BufferRef,
    elements: usize,
};

const Node = union(enum) {
    matmul: Matmul,
    add_bias_relu: AddBiasRelu,
    argmax: ArgMax,

    fn requiresBarrier(self: Node) bool {
        return switch (self) {
            .matmul, .add_bias_relu => true,
            .argmax => false,
        };
    }
};

const Kernel = struct {
    node: Node,
};

const ValueBinding = struct {
    value: *const mlir.Value,
    buffer: BufferRef,
};

const GraphPlan = struct {
    input_count: usize,
    kernels: std.ArrayList(Kernel) = .empty,
    scratch_size: usize = 16,

    fn deinit(self: *GraphPlan, allocator: std.mem.Allocator) void {
        self.kernels.deinit(allocator);
        self.* = undefined;
    }

    fn allocateScratch(
        self: *GraphPlan,
        element_type: KernelType,
        elements: usize,
    ) !BufferRef {
        self.scratch_size = std.mem.alignForward(
            usize,
            self.scratch_size,
            element_type.alignment(),
        );
        const offset = self.scratch_size;
        const byte_count = std.math.mul(
            usize,
            elements,
            element_type.byteSize(),
        ) catch return error.UnsupportedGraph;
        self.scratch_size = std.math.add(
            usize,
            self.scratch_size,
            byte_count,
        ) catch return error.UnsupportedGraph;
        return .{ .scratch = .{
            .offset = offset,
            .element_type = element_type,
            .elements = elements,
        } };
    }
};

pub fn cloneOutputSpecs(allocator: std.mem.Allocator, specs: []const OutputSpec) ![]OutputSpec {
    const result = try allocator.alloc(OutputSpec, specs.len);
    errdefer allocator.free(result);

    var initialized: usize = 0;
    errdefer for (result[0..initialized]) |spec| spec.deinit(allocator);
    for (specs, result) |spec, *cloned| {
        cloned.* = try spec.clone(allocator);
        initialized += 1;
    }
    return result;
}

pub fn deinitOutputSpecs(allocator: std.mem.Allocator, specs: []OutputSpec) void {
    for (specs) |spec| spec.deinit(allocator);
    allocator.free(specs);
}

pub fn parseOutputSpecs(allocator: std.mem.Allocator, program: []const u8) ![]OutputSpec {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    inline for (.{ "func", "stablehlo", "sdy" }) |dialect| {
        mlir.DialectHandle.fromString(dialect).insertDialect(registry);
    }

    const context = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer context.deinit();
    context.loadAllAvailableDialects();

    const module = dialects.stablehlo.deserializePortableArtifact(context, program) catch
        return error.InvalidMlirProgram;
    defer module.deinit();
    return outputSpecsFromModule(allocator, module);
}

pub fn generateKernel(
    allocator: std.mem.Allocator,
    program: []const u8,
) !GeneratedKernel {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    inline for (.{ "func", "stablehlo", "sdy" }) |dialect| {
        mlir.DialectHandle.fromString(dialect).insertDialect(registry);
    }

    const context = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer context.deinit();
    context.loadAllAvailableDialects();

    const module = dialects.stablehlo.deserializePortableArtifact(context, program) catch
        return error.InvalidMlirProgram;
    defer module.deinit();

    const output_specs = try outputSpecsFromModule(allocator, module);
    errdefer deinitOutputSpecs(allocator, output_specs);
    var plan = try graphPlanFromModule(allocator, module, output_specs);
    defer plan.deinit(allocator);
    const source = try renderKernelSource(allocator, plan, output_specs);
    return .{
        .source = source,
        .output_specs = output_specs,
        .scratch_size = plan.scratch_size,
        .input_count = plan.input_count,
    };
}

fn graphPlanFromModule(
    allocator: std.mem.Allocator,
    module: *const mlir.Module,
    outputs: []const OutputSpec,
) !GraphPlan {
    const main = findMainFunction(module) orelse return error.MissingMainFunction;
    if (main.numRegions() != 1) return error.UnsupportedGraph;
    const block = main.region(0).firstBlock() orelse return error.UnsupportedGraph;

    const function_type_attribute = main.attributeByName("function_type") orelse
        return error.UnsupportedGraph;
    const type_attribute = function_type_attribute.isA(mlir.TypeAttribute) orelse
        return error.UnsupportedGraph;
    const function_type = type_attribute.value().isA(mlir.FunctionType) orelse
        return error.UnsupportedGraph;
    const input_count: usize = @intCast(c.mlirFunctionTypeGetNumInputs(function_type.ptr()));

    var plan: GraphPlan = .{ .input_count = input_count };
    errdefer plan.deinit(allocator);
    var bindings: std.ArrayList(ValueBinding) = .empty;
    defer bindings.deinit(allocator);

    var maybe_operation: ?*const mlir.Operation = block.firstOperation();
    while (maybe_operation) |operation| {
        if (std.mem.eql(u8, operation.name(), "stablehlo.dot_general")) {
            if (operation.numOperands() != 2 or
                operation.numResults() != 1)
            {
                return error.UnsupportedGraph;
            }

            const weight_type = rankedTensor(operation.operand(0)) orelse
                return error.UnsupportedGraph;
            const rhs_type = rankedTensor(operation.operand(1)) orelse
                return error.UnsupportedGraph;
            const result_type = rankedTensor(operation.result(0)) orelse
                return error.UnsupportedGraph;
            if (weight_type.rank() != 2 or rhs_type.rank() != 1 or
                result_type.rank() != 1 or
                !isElementType(module.context(), weight_type, c.PJRT_Buffer_Type_F32) or
                !isElementType(module.context(), rhs_type, c.PJRT_Buffer_Type_F32) or
                !isElementType(module.context(), result_type, c.PJRT_Buffer_Type_F32) or
                !isSupportedMatmulDimensions(operation))
            {
                return error.UnsupportedGraph;
            }
            const output_elements = positiveDimension(weight_type, 0) orelse
                return error.UnsupportedGraph;
            const contracting_elements = positiveDimension(weight_type, 1) orelse
                return error.UnsupportedGraph;
            if (positiveDimension(result_type, 0) != output_elements) {
                return error.UnsupportedGraph;
            }
            if (positiveDimension(rhs_type, 0) != contracting_elements) {
                return error.UnsupportedGraph;
            }

            const lhs = resolveBuffer(module, operation.operand(0), bindings.items) orelse
                return error.UnsupportedGraph;
            const rhs = resolveBuffer(module, operation.operand(1), bindings.items) orelse
                return error.UnsupportedGraph;
            const lhs_elements = std.math.mul(
                usize,
                output_elements,
                contracting_elements,
            ) catch return error.UnsupportedGraph;
            if (lhs.elementType() != .f32 or
                lhs.elements() != lhs_elements or
                rhs.elements() != contracting_elements)
            {
                return error.UnsupportedGraph;
            }
            const activation = try plan.allocateScratch(.f32, output_elements);
            try plan.kernels.append(allocator, .{ .node = .{ .matmul = .{
                .lhs = lhs,
                .rhs = rhs,
                .output = activation,
                .m = output_elements,
                .n = 1,
                .k = contracting_elements,
            } } });
            try bindValue(allocator, &bindings, operation.result(0), activation);

            const add = findConsumingOperation(
                operation,
                "stablehlo.add",
                operation.result(0),
            ) orelse return error.UnsupportedGraph;
            const bias_value = if (add.operand(0).eql(operation.result(0)))
                add.operand(1)
            else
                add.operand(0);
            const bias = resolveBuffer(module, bias_value, bindings.items) orelse
                return error.UnsupportedGraph;
            const bias_type = rankedTensor(bias_value) orelse return error.UnsupportedGraph;
            if (bias_type.rank() != 1 or
                positiveDimension(bias_type, 0) != output_elements or
                !isElementType(module.context(), bias_type, c.PJRT_Buffer_Type_F32) or
                bias.elementType() != .f32 or
                bias.elements() != output_elements)
            {
                return error.UnsupportedGraph;
            }

            const maximum = findConsumingOperation(
                add,
                "stablehlo.maximum",
                add.result(0),
            ) orelse return error.UnsupportedGraph;
            try plan.kernels.append(allocator, .{ .node = .{ .add_bias_relu = .{
                .values = activation,
                .bias = bias,
                .rows = 1,
                .columns = output_elements,
            } } });
            try bindValue(allocator, &bindings, maximum.result(0), activation);
        } else if (std.mem.eql(u8, operation.name(), "stablehlo.reduce")) {
            if (operation.numOperands() < 1 or operation.numResults() != 2) {
                return error.UnsupportedGraph;
            }
            const values = resolveBuffer(module, operation.operand(0), bindings.items) orelse
                return error.UnsupportedGraph;
            if (values.elementType() != .f32) return error.UnsupportedGraph;
            const output_index = findReturnedOutput(
                block,
                operation.result(1),
            ) orelse return error.UnsupportedGraph;
            const output = outputBufferRef(outputs, output_index) orelse
                return error.UnsupportedGraph;
            if (output.elements() != 1 or
                (output.elementType() != .u8 and
                    output.elementType() != .i32 and
                    output.elementType() != .u32))
            {
                return error.UnsupportedGraph;
            }
            try plan.kernels.append(allocator, .{ .node = .{ .argmax = .{
                .values = values,
                .output = output,
                .elements = values.elements(),
            } } });
        }
        maybe_operation = nextOperation(operation);
    }

    if (plan.kernels.items.len == 0 or
        std.meta.activeTag(plan.kernels.items[plan.kernels.items.len - 1].node) != .argmax)
    {
        return error.UnsupportedGraph;
    }
    return plan;
}

fn rankedTensor(value: *const mlir.Value) ?*const mlir.RankedTensorType {
    return value.type_().isA(mlir.RankedTensorType);
}

fn positiveDimension(tensor: *const mlir.RankedTensorType, index: usize) ?usize {
    const dimension = tensor.dimension(index);
    return if (dimension >= 0) @intCast(dimension) else null;
}

fn tensorElementCount(tensor: *const mlir.RankedTensorType) ?usize {
    var count: usize = 1;
    for (0..tensor.rank()) |index| {
        count = std.math.mul(
            usize,
            count,
            positiveDimension(tensor, index) orelse return null,
        ) catch return null;
    }
    return count;
}

fn isElementType(
    context: *mlir.Context,
    tensor: *const mlir.RankedTensorType,
    expected: c.PJRT_Buffer_Type,
) bool {
    return (pjrtElementType(context, tensor.elementType()) catch return false) == expected;
}

fn isSupportedMatmulDimensions(operation: *const mlir.Operation) bool {
    const attribute = operation.attributeByName("dot_dimension_numbers") orelse
        return false;
    const dimensions = attribute.isA(
        dialects.stablehlo.DotDimensionNumbersAttribute,
    ) orelse return false;
    return dimensions.getLhsBatchingDimensionsSize() == 0 and
        dimensions.getRhsBatchingDimensionsSize() == 0 and
        dimensions.getLhsContractingDimensionsSize() == 1 and
        dimensions.getRhsContractingDimensionsSize() == 1 and
        dimensions.getLhsContractingDimensionsElem(0) == 1 and
        dimensions.getRhsContractingDimensionsElem(0) == 0;
}

fn bindValue(
    allocator: std.mem.Allocator,
    bindings: *std.ArrayList(ValueBinding),
    value: *const mlir.Value,
    buffer: BufferRef,
) !void {
    try bindings.append(allocator, .{ .value = value, .buffer = buffer });
}

fn resolveBuffer(
    module: *const mlir.Module,
    value: *const mlir.Value,
    bindings: []const ValueBinding,
) ?BufferRef {
    var binding_index = bindings.len;
    while (binding_index > 0) {
        binding_index -= 1;
        const binding = bindings[binding_index];
        if (binding.value.eql(value)) return binding.buffer;
    }

    if (value.isA(mlir.BlockArgument)) |argument| {
        const tensor = rankedTensor(value) orelse return null;
        return .{ .argument = .{
            .index = argument.number(),
            .element_type = kernelType(
                module.context(),
                tensor.elementType(),
            ) orelse return null,
            .elements = tensorElementCount(tensor) orelse return null,
        } };
    }

    const result = value.isA(mlir.OpResult) orelse return null;
    const owner = result.owner();
    if (owner.numOperands() != 1 or
        (!std.mem.eql(u8, owner.name(), "stablehlo.reshape") and
            !std.mem.eql(u8, owner.name(), "stablehlo.convert")))
    {
        return null;
    }
    return resolveBuffer(module, owner.operand(0), bindings);
}

fn outputBufferRef(outputs: []const OutputSpec, index: usize) ?BufferRef {
    if (index >= outputs.len) return null;
    const output = outputs[index];
    var elements: usize = 1;
    for (output.dims) |dimension| {
        if (dimension < 0) return null;
        elements = std.math.mul(usize, elements, @intCast(dimension)) catch
            return null;
    }
    return .{ .output = .{
        .index = index,
        .element_type = kernelTypeFromPjrt(output.element_type) orelse return null,
        .elements = elements,
    } };
}

fn kernelType(
    context: *mlir.Context,
    element_type: *const mlir.Type,
) ?KernelType {
    return kernelTypeFromPjrt(pjrtElementType(context, element_type) catch
        return null);
}

fn kernelTypeFromPjrt(element_type: c.PJRT_Buffer_Type) ?KernelType {
    return switch (element_type) {
        c.PJRT_Buffer_Type_U8 => .u8,
        c.PJRT_Buffer_Type_S32 => .i32,
        c.PJRT_Buffer_Type_U32 => .u32,
        c.PJRT_Buffer_Type_F32 => .f32,
        else => null,
    };
}

fn findReturnedOutput(
    block: *const mlir.Block,
    value: *const mlir.Value,
) ?usize {
    const terminator = block.terminator() orelse return null;
    if (!std.mem.eql(u8, terminator.name(), "func.return")) return null;
    for (0..terminator.numOperands()) |index| {
        if (isDerivedFrom(terminator.operand(index), value)) return index;
    }
    return null;
}

fn isDerivedFrom(value: *const mlir.Value, source: *const mlir.Value) bool {
    if (value.eql(source)) return true;
    const result = value.isA(mlir.OpResult) orelse return false;
    const owner = result.owner();
    if (owner.numOperands() != 1 or
        (!std.mem.eql(u8, owner.name(), "stablehlo.reshape") and
            !std.mem.eql(u8, owner.name(), "stablehlo.convert") and
            !std.mem.eql(u8, owner.name(), "stablehlo.broadcast_in_dim")))
    {
        return false;
    }
    return isDerivedFrom(owner.operand(0), source);
}

fn findConsumingOperation(
    after: *const mlir.Operation,
    name: []const u8,
    value: *const mlir.Value,
) ?*const mlir.Operation {
    var maybe_operation = nextOperation(after);
    while (maybe_operation) |operation| {
        if (std.mem.eql(u8, operation.name(), name)) {
            for (0..operation.numOperands()) |index| {
                if (operation.operand(index).eql(value)) return operation;
            }
        }
        if (std.mem.eql(u8, operation.name(), "stablehlo.dot_general") or
            std.mem.eql(u8, operation.name(), "func.return"))
        {
            return null;
        }
        maybe_operation = nextOperation(operation);
    }
    return null;
}

fn nextOperation(operation: *const mlir.Operation) ?*const mlir.Operation {
    const next = c.mlirOperationGetNextInBlock(operation.ptr());
    return if (next.ptr) |ptr| @ptrCast(ptr) else null;
}

fn renderKernelSource(
    allocator: std.mem.Allocator,
    plan: GraphPlan,
    outputs: []const OutputSpec,
) ![]u8 {
    var source: std.Io.Writer.Allocating = try .initCapacity(allocator, 4096);
    errdefer source.deinit();
    const writer = &source.writer;
    const scratch_index = plan.input_count + outputs.len;
    const expected_buffers = scratch_index + 1;

    try writer.writeAll(
        \\const kernels = @import("kernels.zig");
        \\
        \\fn constBuffer(comptime T: type, buffers: [*]*const anyopaque, index: usize) [*]const T {
        \\    return @ptrCast(@alignCast(buffers[index]));
        \\}
        \\
        \\fn mutableBuffer(comptime T: type, buffers: [*]*const anyopaque, index: usize) [*]T {
        \\    return @ptrCast(@alignCast(@constCast(buffers[index])));
        \\}
        \\
        \\fn scratchBuffer(comptime T: type, scratch: [*]u8, offset: usize) [*]T {
        \\    return @ptrCast(@alignCast(scratch + offset));
        \\}
        \\
    );
    try writer.print(
        \\export fn main(buffers: [*]*const anyopaque, buffer_len: usize) callconv(.nvptx_kernel) void {{
        \\    if (buffer_len != {d}) return;
        \\    const scratch = mutableBuffer(u8, buffers, {d});
        \\    const barrier: *kernels.GridBarrier = @ptrCast(@alignCast(scratch));
        \\
    , .{
        expected_buffers,
        scratch_index,
    });

    for (plan.kernels.items) |kernel| {
        try renderNode(writer, kernel.node, plan.input_count);
        if (kernel.node.requiresBarrier()) {
            try writer.writeAll("    kernels.gridBarrier(barrier);\n");
        }
    }
    try writer.writeAll("}\n");
    return source.toOwnedSlice();
}

fn renderNode(
    writer: *std.Io.Writer,
    node: Node,
    input_count: usize,
) std.Io.Writer.Error!void {
    switch (node) {
        .matmul => |matmul| {
            try writer.print(
                "    kernels.matmulF32({s}, ",
                .{matmul.rhs.elementType().zigName()},
            );
            try renderBufferRef(writer, matmul.lhs, input_count, false);
            try writer.writeAll(", ");
            try renderBufferRef(writer, matmul.rhs, input_count, false);
            try writer.writeAll(", ");
            try renderBufferRef(writer, matmul.output, input_count, true);
            try writer.print(
                ", {d}, {d}, {d});\n",
                .{ matmul.m, matmul.n, matmul.k },
            );
        },
        .add_bias_relu => |add_bias_relu| {
            try writer.writeAll("    kernels.addBiasReluF32(");
            try renderBufferRef(writer, add_bias_relu.values, input_count, true);
            try writer.writeAll(", ");
            try renderBufferRef(writer, add_bias_relu.bias, input_count, false);
            try writer.print(
                ", {d}, {d});\n",
                .{ add_bias_relu.rows, add_bias_relu.columns },
            );
        },
        .argmax => |argmax| {
            try writer.print(
                "    kernels.argMaxF32({s}, ",
                .{argmax.output.elementType().zigName()},
            );
            try renderBufferRef(writer, argmax.values, input_count, false);
            try writer.print(", {d}, &", .{argmax.elements});
            try renderBufferRef(writer, argmax.output, input_count, true);
            try writer.writeAll("[0]);\n");
        },
    }
}

fn renderBufferRef(
    writer: *std.Io.Writer,
    buffer_ref: BufferRef,
    input_count: usize,
    mutable: bool,
) std.Io.Writer.Error!void {
    switch (buffer_ref) {
        .argument => |buffer| try writer.print(
            "{s}Buffer({s}, buffers, {d})",
            .{
                if (mutable) "mutable" else "const",
                buffer.element_type.zigName(),
                buffer.index,
            },
        ),
        .output => |buffer| try writer.print(
            "{s}Buffer({s}, buffers, {d})",
            .{
                if (mutable) "mutable" else "const",
                buffer.element_type.zigName(),
                input_count + buffer.index,
            },
        ),
        .scratch => |buffer| try writer.print(
            "scratchBuffer({s}, scratch, {d})",
            .{ buffer.element_type.zigName(), buffer.offset },
        ),
    }
}

fn outputSpecsFromModule(allocator: std.mem.Allocator, module: *const mlir.Module) ![]OutputSpec {
    const main = findMainFunction(module) orelse return error.MissingMainFunction;
    const function_type_attribute = main.attributeByName("function_type") orelse
        return error.MissingMainFunction;
    const type_attribute = function_type_attribute.isA(mlir.TypeAttribute) orelse
        return error.MissingMainFunction;
    const function_type = type_attribute.value().isA(mlir.FunctionType) orelse
        return error.MissingMainFunction;

    const output_count: usize = @intCast(c.mlirFunctionTypeGetNumResults(function_type.ptr()));
    const outputs = try allocator.alloc(OutputSpec, output_count);
    errdefer allocator.free(outputs);

    var initialized: usize = 0;
    errdefer for (outputs[0..initialized]) |spec| spec.deinit(allocator);
    for (outputs, 0..) |*output, index| {
        const result_type: *const mlir.Type = @ptrCast(
            c.mlirFunctionTypeGetResult(function_type.ptr(), @intCast(index)).ptr,
        );
        const tensor_type = result_type.isA(mlir.RankedTensorType) orelse
            return error.UnsupportedOutputType;

        const dims = try allocator.alloc(i64, tensor_type.rank());
        errdefer allocator.free(dims);
        for (dims, 0..) |*dim, dimension_index| {
            dim.* = tensor_type.dimension(dimension_index);
            if (dim.* < 0) return error.UnsupportedOutputType;
        }
        output.* = .{
            .element_type = try pjrtElementType(module.context(), tensor_type.elementType()),
            .dims = dims,
        };
        initialized += 1;
    }
    return outputs;
}

fn findMainFunction(module: *const mlir.Module) ?*const mlir.Operation {
    var maybe_operation: ?*const mlir.Operation = module.body().firstOperation();
    while (maybe_operation) |operation| {
        if (std.mem.eql(u8, operation.name(), "func.func")) {
            if (operation.attributeByName("sym_name")) |name_attribute| {
                if (name_attribute.isA(mlir.StringAttribute)) |name| {
                    if (std.mem.eql(u8, name.value(), "main")) return operation;
                }
            }
        }

        const next = c.mlirOperationGetNextInBlock(operation.ptr());
        maybe_operation = if (next.ptr) |ptr| @ptrCast(ptr) else null;
    }
    return null;
}

fn pjrtElementType(context: *mlir.Context, element_type: *const mlir.Type) !c.PJRT_Buffer_Type {
    const mapping = .{
        .{ c.PJRT_Buffer_Type_PRED, mlir.Type.int(context, .i1) },
        .{ c.PJRT_Buffer_Type_S2, mlir.Type.int(context, .i2) },
        .{ c.PJRT_Buffer_Type_S4, mlir.Type.int(context, .i4) },
        .{ c.PJRT_Buffer_Type_S8, mlir.Type.int(context, .i8) },
        .{ c.PJRT_Buffer_Type_S16, mlir.Type.int(context, .i16) },
        .{ c.PJRT_Buffer_Type_S32, mlir.Type.int(context, .i32) },
        .{ c.PJRT_Buffer_Type_S64, mlir.Type.int(context, .i64) },
        .{ c.PJRT_Buffer_Type_U2, mlir.Type.int(context, .u2) },
        .{ c.PJRT_Buffer_Type_U4, mlir.Type.int(context, .u4) },
        .{ c.PJRT_Buffer_Type_U8, mlir.Type.int(context, .u8) },
        .{ c.PJRT_Buffer_Type_U16, mlir.Type.int(context, .u16) },
        .{ c.PJRT_Buffer_Type_U32, mlir.Type.int(context, .u32) },
        .{ c.PJRT_Buffer_Type_U64, mlir.Type.int(context, .u64) },
        .{ c.PJRT_Buffer_Type_F4E2M1FN, mlir.Type.float(context, .f4e2m1fn) },
        .{ c.PJRT_Buffer_Type_F8E3M4, mlir.Type.float(context, .f8e3m4) },
        .{ c.PJRT_Buffer_Type_F8E4M3, mlir.Type.float(context, .f8e4m3) },
        .{ c.PJRT_Buffer_Type_F8E4M3B11FNUZ, mlir.Type.float(context, .f8e4m3b11fnuz) },
        .{ c.PJRT_Buffer_Type_F8E4M3FN, mlir.Type.float(context, .f8e4m3fn) },
        .{ c.PJRT_Buffer_Type_F8E4M3FNUZ, mlir.Type.float(context, .f8e4m3fnuz) },
        .{ c.PJRT_Buffer_Type_F8E5M2, mlir.Type.float(context, .f8e5m2) },
        .{ c.PJRT_Buffer_Type_F8E5M2FNUZ, mlir.Type.float(context, .f8e5m2fnuz) },
        .{ c.PJRT_Buffer_Type_F8E8M0FNU, mlir.Type.float(context, .f8e8m0fnu) },
        .{ c.PJRT_Buffer_Type_BF16, mlir.Type.float(context, .bf16) },
        .{ c.PJRT_Buffer_Type_F16, mlir.Type.float(context, .f16) },
        .{ c.PJRT_Buffer_Type_F32, mlir.Type.float(context, .f32) },
        .{ c.PJRT_Buffer_Type_F64, mlir.Type.float(context, .f64) },
        .{ c.PJRT_Buffer_Type_C64, mlir.Type.complex(context, .c64) },
        .{ c.PJRT_Buffer_Type_C128, mlir.Type.complex(context, .c128) },
    };
    inline for (mapping) |entry| {
        if (element_type.eql(entry[1])) return entry[0];
    }
    return error.UnsupportedOutputType;
}

test "read output specifications from main function" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    mlir.DialectHandle.fromString("func").insertDialect(registry);

    const context = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer context.deinit();
    context.loadAllAvailableDialects();

    const module = try mlir.Module.parse(
        context,
        "module { func.func private @main() -> (tensor<2x3xf32>, tensor<4xui32>) }",
    );
    defer module.deinit();

    const outputs = try outputSpecsFromModule(std.testing.allocator, module);
    defer deinitOutputSpecs(std.testing.allocator, outputs);
    try std.testing.expectEqual(@as(usize, 2), outputs.len);
    try std.testing.expectEqual(
        @as(c.PJRT_Buffer_Type, c.PJRT_Buffer_Type_F32),
        outputs[0].element_type,
    );
    try std.testing.expectEqualSlices(i64, &.{ 2, 3 }, outputs[0].dims);
    try std.testing.expectEqual(
        @as(c.PJRT_Buffer_Type, c.PJRT_Buffer_Type_U32),
        outputs[1].element_type,
    );
    try std.testing.expectEqualSlices(i64, &.{4}, outputs[1].dims);

    var portable_artifact: std.Io.Writer.Allocating = try .initCapacity(
        std.testing.allocator,
        4096,
    );
    defer portable_artifact.deinit();
    try dialects.stablehlo.serializePortableArtifact2(
        module,
        dialects.stablehlo.minimumVersion(),
        &portable_artifact.writer,
    );

    const portable_outputs = try parseOutputSpecs(
        std.testing.allocator,
        portable_artifact.written(),
    );
    defer deinitOutputSpecs(std.testing.allocator, portable_outputs);
    try std.testing.expectEqual(@as(usize, 2), portable_outputs.len);
    try std.testing.expectEqualSlices(i64, &.{ 2, 3 }, portable_outputs[0].dims);
    try std.testing.expectEqualSlices(i64, &.{4}, portable_outputs[1].dims);
}

test "lower dense StableHLO graph to composable Zig sub-kernels" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    inline for (.{ "func", "stablehlo" }) |dialect| {
        mlir.DialectHandle.fromString(dialect).insertDialect(registry);
    }

    const context = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer context.deinit();
    context.loadAllAvailableDialects();

    const module = try mlir.Module.parse(context,
        \\module {
        \\  func.func public @main(%arg0: tensor<2x3xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2x2xf32>, %arg3: tensor<2xf32>, %arg4: tensor<3xui8>) -> tensor<1xui8> {
        \\    %c = stablehlo.constant dense<0> : tensor<i32>
        \\    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
        \\    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        \\    %0 = stablehlo.convert %arg4 : (tensor<3xui8>) -> tensor<3xf32>
        \\    %1 = stablehlo.dot_general %arg0, %0, contracting_dims = [1] x [0] : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2xf32>
        \\    %2 = stablehlo.add %1, %arg1 : tensor<2xf32>
        \\    %3 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<2xf32>
        \\    %4 = stablehlo.maximum %2, %3 : tensor<2xf32>
        \\    %5 = stablehlo.dot_general %arg2, %4, contracting_dims = [1] x [0] : (tensor<2x2xf32>, tensor<2xf32>) -> tensor<2xf32>
        \\    %6 = stablehlo.add %5, %arg3 : tensor<2xf32>
        \\    %7 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<2xf32>
        \\    %8 = stablehlo.maximum %6, %7 : tensor<2xf32>
        \\    %9 = stablehlo.iota dim = 0 : tensor<2xi32>
        \\    %10:2 = stablehlo.reduce(%8 init: %cst), (%9 init: %c) across dimensions = [0] : (tensor<2xf32>, tensor<2xi32>, tensor<f32>, tensor<i32>) -> (tensor<f32>, tensor<i32>)
        \\     reducer(%lhs_value: tensor<f32>, %rhs_value: tensor<f32>) (%lhs_index: tensor<i32>, %rhs_index: tensor<i32>) {
        \\      %is_greater = stablehlo.compare GT, %lhs_value, %rhs_value, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
        \\      %value = stablehlo.select %is_greater, %lhs_value, %rhs_value : tensor<i1>, tensor<f32>
        \\      %index = stablehlo.select %is_greater, %lhs_index, %rhs_index : tensor<i1>, tensor<i32>
        \\      stablehlo.return %value, %index : tensor<f32>, tensor<i32>
        \\    }
        \\    %11 = stablehlo.broadcast_in_dim %10#1, dims = [] : (tensor<i32>) -> tensor<1xi32>
        \\    %12 = stablehlo.convert %11 : (tensor<1xi32>) -> tensor<1xui8>
        \\    return %12 : tensor<1xui8>
        \\  }
        \\}
    );
    defer module.deinit();

    const plan_outputs = try outputSpecsFromModule(std.testing.allocator, module);
    defer deinitOutputSpecs(std.testing.allocator, plan_outputs);
    var plan = try graphPlanFromModule(std.testing.allocator, module, plan_outputs);
    defer plan.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 5), plan.kernels.items.len);
    try std.testing.expectEqual(.matmul, std.meta.activeTag(plan.kernels.items[0].node));
    try std.testing.expectEqual(.add_bias_relu, std.meta.activeTag(plan.kernels.items[1].node));
    try std.testing.expectEqual(.matmul, std.meta.activeTag(plan.kernels.items[2].node));
    try std.testing.expectEqual(.add_bias_relu, std.meta.activeTag(plan.kernels.items[3].node));
    try std.testing.expectEqual(.argmax, std.meta.activeTag(plan.kernels.items[4].node));

    var artifact: std.Io.Writer.Allocating = try .initCapacity(
        std.testing.allocator,
        4096,
    );
    defer artifact.deinit();
    try dialects.stablehlo.serializePortableArtifact2(
        module,
        dialects.stablehlo.minimumVersion(),
        &artifact.writer,
    );

    var generated = try generateKernel(std.testing.allocator, artifact.written());
    defer generated.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 5), generated.input_count);
    try std.testing.expectEqual(@as(usize, 32), generated.scratch_size);
    try std.testing.expect(std.mem.indexOf(
        u8,
        generated.source,
        "kernels.matmulF32(u8, constBuffer(f32, buffers, 0), constBuffer(u8, buffers, 4), scratchBuffer(f32, scratch, 16), 2, 1, 3);",
    ) != null);
    try std.testing.expect(std.mem.indexOf(
        u8,
        generated.source,
        "kernels.matmulF32(f32, constBuffer(f32, buffers, 2), scratchBuffer(f32, scratch, 16), scratchBuffer(f32, scratch, 24), 2, 1, 2);",
    ) != null);
    try std.testing.expect(std.mem.indexOf(
        u8,
        generated.source,
        "kernels.addBiasReluF32(scratchBuffer(f32, scratch, 16), constBuffer(f32, buffers, 1), 1, 2);",
    ) != null);
    try std.testing.expect(std.mem.indexOf(
        u8,
        generated.source,
        "kernels.addBiasF32",
    ) == null);
    try std.testing.expect(std.mem.indexOf(
        u8,
        generated.source,
        "kernels.reluF32",
    ) == null);
    try std.testing.expect(std.mem.indexOf(
        u8,
        generated.source,
        "kernels.argMaxF32(u8, scratchBuffer(f32, scratch, 24), 2, &mutableBuffer(u8, buffers, 5)[0]);",
    ) != null);

    const ptx = try compileGenerated(std.testing.allocator, generated.source);
    defer std.testing.allocator.free(ptx);
    try std.testing.expect(std.mem.indexOf(u8, ptx, ".visible .entry main(") != null);
    try std.testing.expect(fusedBiasReluLoops(ptx, 2));

    const cuda = @import("cuda.zig");
    var cuda_client = cuda.Client.init() catch |err| switch (err) {
        error.DriverUnavailable,
        error.InitializationFailed,
        error.DeviceUnavailable,
        => return error.SkipZigTest,
        else => return err,
    };
    defer cuda_client.deinit();

    const weight_0 = [_]f32{ 1, 2, 3, 3, 2, 1 };
    const bias_0 = [_]f32{ 0, 0 };
    const weight_1 = [_]f32{ 1, 0, 0, 2 };
    const bias_1 = [_]f32{ 0, 0 };
    const input = [_]u8{ 1, 2, 3 };
    const host_inputs = .{ weight_0, bias_0, weight_1, bias_1, input };

    var allocations: [host_inputs.len]cuda.Allocation = undefined;
    var initialized: usize = 0;
    defer for (allocations[0..initialized]) |allocation| cuda_client.free(allocation);
    inline for (host_inputs, 0..) |host_input, index| {
        allocations[index] = try cuda_client.allocate(@sizeOf(@TypeOf(host_input)));
        initialized += 1;
        try cuda_client.copyHostToDevice(
            allocations[index],
            0,
            std.mem.asBytes(&host_input),
        );
    }

    const output = try cuda_client.allocate(@sizeOf(u8));
    defer cuda_client.free(output);
    try cuda_client.zero(output);
    const scratch = try cuda_client.allocate(generated.scratch_size);
    defer cuda_client.free(scratch);
    try cuda_client.zero(scratch);

    var pointers: [host_inputs.len + 2]cuda.DevicePtr = undefined;
    for (allocations, pointers[0..host_inputs.len]) |allocation, *pointer| {
        pointer.* = allocation.ptr;
    }
    pointers[host_inputs.len] = output.ptr;
    pointers[host_inputs.len + 1] = scratch.ptr;
    const pointer_table = try cuda_client.allocate(@sizeOf(@TypeOf(pointers)));
    defer cuda_client.free(pointer_table);
    try cuda_client.copyHostToDevice(pointer_table, 0, std.mem.asBytes(&pointers));

    const kernel = try cuda_client.loadKernel(ptx, "main");
    defer cuda_client.unloadKernel(kernel);
    const Parameters = extern struct {
        buffers: cuda.DevicePtr,
        buffer_len: usize,
    };
    const parameters: Parameters = .{
        .buffers = pointer_table.ptr,
        .buffer_len = pointers.len,
    };
    try cuda_client.launch(kernel, std.mem.asBytes(&parameters));

    var actual: u8 = undefined;
    try cuda_client.copyDeviceToHost(std.mem.asBytes(&actual), output, 0);
    try std.testing.expectEqual(@as(u8, 1), actual);
}

fn fusedBiasReluLoops(ptx: []const u8, expected_count: usize) bool {
    if (std.mem.indexOf(u8, ptx, "%kernels.addBiasF32.exit") != null or
        std.mem.indexOf(u8, ptx, "%kernels.reluF32.exit") != null)
    {
        return false;
    }

    const loop_header = "// =>This Inner Loop Header: Depth=1";
    const fused_exit = "%kernels.addBiasReluF32.exit";
    var remaining = ptx;
    var count: usize = 0;
    while (std.mem.indexOf(u8, remaining, fused_exit)) |exit_index| {
        const header_index = std.mem.lastIndexOf(
            u8,
            remaining[0..exit_index],
            loop_header,
        ) orelse return false;
        const body = remaining[header_index..exit_index];
        if (std.mem.indexOf(u8, body, "add.rn.f32") == null or
            std.mem.indexOf(u8, body, "max.f32") == null)
        {
            return false;
        }
        count += 1;
        remaining = remaining[exit_index + fused_exit.len ..];
    }
    return count == expected_count;
}

pub fn compile(allocator: std.mem.Allocator) ![:0]u8 {
    return compileSource(allocator, null);
}

pub fn compileGenerated(
    allocator: std.mem.Allocator,
    source: []const u8,
) ![:0]u8 {
    return compileSource(allocator, source);
}

fn compileSource(
    allocator: std.mem.Allocator,
    generated_source: ?[]const u8,
) ![:0]u8 {
    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const executable_path = try std.process.executablePathAlloc(io, allocator);
    defer allocator.free(executable_path);

    var runfiles_instance = try runfiles.Runfiles.create(.{
        .allocator = allocator,
        .io = io,
        .argv0 = executable_path,
        .directory = if (std.c.getenv("RUNFILES_DIR")) |value| std.mem.span(value) else null,
        .manifest = if (std.c.getenv("RUNFILES_MANIFEST_FILE")) |value| std.mem.span(value) else null,
    }) orelse return error.RunfilesUnavailable;
    defer runfiles_instance.deinit(allocator);
    const r = runfiles_instance.withSourceRepo(bazel_builtin.current_repository);

    const config_path = try r.rlocationAlloc(allocator, zig_runtime_config) orelse
        return error.ToolchainConfigUnavailable;
    defer allocator.free(config_path);
    const config = try std.Io.Dir.cwd().readFileAlloc(io, config_path, allocator, .limited(16 * 1024));
    defer allocator.free(config);
    var config_lines = std.mem.tokenizeScalar(u8, config, '\n');
    const zig_runfile = config_lines.next() orelse return error.ToolchainConfigUnavailable;
    const zig_lib_runfile = config_lines.next() orelse return error.ToolchainConfigUnavailable;
    const kernel_runfile = config_lines.next() orelse return error.ToolchainConfigUnavailable;

    const zig_path = try r.rlocationAlloc(allocator, zig_runfile) orelse
        return error.ZigExecutableUnavailable;
    defer allocator.free(zig_path);
    const zig_lib_path = try r.rlocationAlloc(allocator, zig_lib_runfile) orelse
        return error.ZigLibUnavailable;
    defer allocator.free(zig_lib_path);
    const kernel_path = try r.rlocationAlloc(allocator, kernel_runfile) orelse
        return error.KernelSourceUnavailable;
    defer allocator.free(kernel_path);
    const kernel_directory = std.fs.path.dirname(kernel_path) orelse
        return error.KernelLibraryUnavailable;
    const kernels_path = try std.Io.Dir.path.join(
        allocator,
        &.{ kernel_directory, "kernels.zig" },
    );
    defer allocator.free(kernels_path);

    const kernel_source = if (generated_source) |source|
        source
    else
        try std.Io.Dir.cwd().readFileAlloc(
            io,
            kernel_path,
            allocator,
            .limited(4 * 1024 * 1024),
        );
    defer if (generated_source == null) allocator.free(kernel_source);
    const kernels_source = std.Io.Dir.cwd().readFileAlloc(
        io,
        kernels_path,
        allocator,
        .limited(4 * 1024 * 1024),
    ) catch return error.KernelLibraryUnavailable;
    defer allocator.free(kernels_source);

    const tmp_root_path = if (std.c.getenv("TMPDIR")) |value| std.mem.span(value) else "/tmp";
    var tmp_root = try std.Io.Dir.openDir(.cwd(), io, tmp_root_path, .{});
    defer tmp_root.close(io);

    var dir_name_buffer: [128]u8 = undefined;
    const dir_name = try std.fmt.bufPrint(
        &dir_name_buffer,
        "zml-cudaz-{d}",
        .{std.Io.Timestamp.now(io, .real).nanoseconds},
    );
    var tmp_dir = try tmp_root.createDirPathOpen(io, dir_name, .{ .permissions = .fromMode(0o700) });
    defer tmp_root.deleteTree(io, dir_name) catch {};
    defer tmp_dir.close(io);

    var tmp_path_buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const tmp_path_len = try tmp_dir.realPath(io, &tmp_path_buffer);
    const tmp_path = tmp_path_buffer[0..tmp_path_len];

    const kernel_file = try tmp_dir.createFile(io, "kernel.zig", .{});
    defer kernel_file.close(io);
    try kernel_file.writePositionalAll(io, kernel_source, 0);
    const kernels_file = try tmp_dir.createFile(io, "kernels.zig", .{});
    defer kernels_file.close(io);
    try kernels_file.writePositionalAll(io, kernels_source, 0);

    try runZig(allocator, io, tmp_path, &.{
        zig_path,
        "build-obj",
        "kernel.zig",
        "-target",
        "nvptx64-cuda",
        "-mcpu",
        "sm_50+ptx63",
        "-O",
        "ReleaseFast",
        "-fno-incremental",
        "-fno-entry",
        "-fno-emit-bin",
        "-femit-llvm-ir=kernel.ll",
        "--zig-lib-dir",
        zig_lib_path,
        "--cache-dir",
        "cache",
        "--global-cache-dir",
        "global-cache",
    });

    const llvm_ir = try tmp_dir.readFileAlloc(io, "kernel.ll", allocator, .limited(4 * 1024 * 1024));
    defer allocator.free(llvm_ir);
    const fixed_llvm_ir = try rewriteKernelIr(allocator, llvm_ir);
    defer allocator.free(fixed_llvm_ir);

    const fixed_ir_file = try tmp_dir.createFile(io, "kernel.fixed.ll", .{});
    defer fixed_ir_file.close(io);
    try fixed_ir_file.writePositionalAll(io, fixed_llvm_ir, 0);

    try runZig(allocator, io, tmp_path, &.{
        zig_path,
        "cc",
        "-target",
        "nvptx64-cuda",
        "-Xclang",
        "-target-cpu",
        "-Xclang",
        "sm_50",
        "-Xclang",
        "-target-feature",
        "-Xclang",
        "+ptx63",
        "-S",
        "kernel.fixed.ll",
        "-o",
        "kernel.ptx",
    });

    const ptx = try tmp_dir.readFileAllocOptions(
        io,
        "kernel.ptx",
        allocator,
        .limited(4 * 1024 * 1024),
        .of(u8),
        0,
    );
    if (std.c.getenv("ZML_CUDAZ_DUMP_PTX") != null) {
        log.info("generated PTX:\n{s}", .{ptx});
    }
    return ptx;
}

fn runZig(
    allocator: std.mem.Allocator,
    io: std.Io,
    cwd: []const u8,
    argv: []const []const u8,
) !void {
    const local_cache = try std.Io.Dir.path.join(allocator, &.{ cwd, "cache" });
    defer allocator.free(local_cache);
    const global_cache = try std.Io.Dir.path.join(allocator, &.{ cwd, "global-cache" });
    defer allocator.free(global_cache);
    var environ = std.process.Environ.Map.init(allocator);
    defer environ.deinit();
    try environ.put("ZIG_LOCAL_CACHE_DIR", local_cache);
    try environ.put("ZIG_GLOBAL_CACHE_DIR", global_cache);

    const result = std.process.run(allocator, io, .{
        .argv = argv,
        .cwd = .{ .path = cwd },
        .environ_map = &environ,
        .stdout_limit = .limited(1024 * 1024),
        .stderr_limit = .limited(1024 * 1024),
    }) catch return error.ZigCompilationFailed;
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    switch (result.term) {
        .exited => |code| if (code == 0) return,
        else => {},
    }
    log.err("Zig PTX compilation failed:\n{s}", .{result.stderr});
    return error.ZigCompilationFailed;
}

fn rewriteKernelIr(allocator: std.mem.Allocator, llvm_ir: []const u8) ![]u8 {
    const alias_start = std.mem.indexOf(u8, llvm_ir, "@main = alias") orelse
        return error.InvalidKernelIr;
    const alias_end = std.mem.indexOfPos(u8, llvm_ir, alias_start, "\n") orelse
        return error.InvalidKernelIr;
    const kernel_declaration = "define private ptx_kernel void @kernel.main";
    if (std.mem.indexOf(u8, llvm_ir, kernel_declaration) == null) return error.InvalidKernelIr;

    const without_alias = try std.mem.concat(allocator, u8, &.{
        llvm_ir[0..alias_start],
        llvm_ir[alias_end + 1 ..],
    });
    defer allocator.free(without_alias);
    const renamed = try std.mem.replaceOwned(
        u8,
        allocator,
        without_alias,
        kernel_declaration,
        "define ptx_kernel void @main",
    );
    defer allocator.free(renamed);
    return try std.mem.replaceOwned(u8, allocator, renamed, " unnamed_addr", "");
}

test "rewrite exported Zig kernel IR for the NVPTX backend" {
    const input =
        \\@main = alias void (ptr, i64), ptr @kernel.main
        \\
        \\define private ptx_kernel void @kernel.main(ptr %0, i64 %1) unnamed_addr {
        \\  ret void
        \\}
    ;
    const actual = try rewriteKernelIr(std.testing.allocator, input);
    defer std.testing.allocator.free(actual);
    try std.testing.expectEqualStrings(
        \\
        \\define ptx_kernel void @main(ptr %0, i64 %1) {
        \\  ret void
        \\}
    ,
        actual,
    );
}

test "Bazel Zig toolchain compiles the cudaz kernel to PTX" {
    const ptx = try compile(std.testing.allocator);
    defer std.testing.allocator.free(ptx);
    try std.testing.expect(std.mem.indexOf(u8, ptx, ".visible .entry main(") != null);
}
