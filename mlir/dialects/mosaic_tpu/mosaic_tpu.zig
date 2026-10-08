const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");
const stdx = @import("stdx");

/// Mosaic IR version stored as `stable_mosaic.version`, matching
/// `xla/mosaic/dialect/tpu/transforms/serde.h:kVersion`. This is distinct from
/// the custom-call JSON `serialization_format`, which remains 1.
pub const SERDE_VERSION: i32 = 18;

/// Register the `mosaic-serde` pass so `PassManager.parse` can pick it up by
/// name. Idempotent — safe to call repeatedly. Required before running the
/// `mosaic-serde{serialize=true}` pipeline that produces the bytecode embedded
/// in `tpu_custom_call`'s backend_config.
pub fn registerMosaicSerdePass() void {
    c.mlirTpuRegisterMosaicSerdePass();
}

// =============================================================================
// Types — !tpu.semaphore, !tpu.dma_semaphore, !tpu.float8_exmy<...>
// =============================================================================

/// `!tpu.semaphore`.
pub const SemaphoreType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsATpuSemaphore;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "semaphore";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirTpuSemaphoreTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!tpu.semaphore` element type.
pub fn semaphoreType(ctx: *mlir.Context) *const mlir.Type {
    const ty = SemaphoreType.get(ctx, .{}) catch unreachable;
    return ty.type_();
}

/// `!tpu.dma_semaphore`.
pub const DMASemaphoreType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsATpuDMASemaphore;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "dma_semaphore";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirTpuDMASemaphoreTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// Build `!tpu.dma_semaphore`.
pub fn dmaSemaphoreType(ctx: *mlir.Context) *const mlir.Type {
    const ty = DMASemaphoreType.get(ctx, .{}) catch unreachable;
    return ty.type_();
}

/// `!tpu.float8_exmy<UnderlyingType>` — Mosaic Float8 type.
pub const Float8EXMYType = opaque {
    const M = mlir.Methods(Float8EXMYType, c.MlirType);

    pub const isAFn = c.mlirTpuIsAFloat8EXMYType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);
    pub const isA = M.isA;

    pub fn get(ctx: *mlir.Context, underlying: *const mlir.Type) *const Float8EXMYType {
        return @ptrCast(c.mlirTpuFloat8EXMYTypeGet(ctx.ptr(), underlying.ptr()).ptr);
    }

    pub fn underlyingType(self: *const Float8EXMYType) *const mlir.Type {
        return @ptrCast(c.mlirTpuFloat8EXMYTypeGetUnderlyingType(self.ptr()).ptr);
    }
};

pub fn float8ExmyType(ctx: *mlir.Context, underlying: *const mlir.Type) *const mlir.Type {
    return @ptrCast(Float8EXMYType.get(ctx, underlying));
}

// =============================================================================
// Attributes
// =============================================================================

/// Reduction kind for `tpu.all_reduce`, `tpu.reduce_index`, `tpu.scan`.
pub const ReductionKind = enum(u32) {
    sum = 0,
    max = 1,
    min = 2,
    arg_max = 3,
    arg_min = 4,
    find_first_set = 5,
    maxf = 6,
    minf = 7,
    maxsi = 8,
    minsi = 9,
    maxui = 10,
    minui = 11,

    pub fn attribute(self: ReductionKind, ctx: *mlir.Context) *const mlir.Attribute {
        const attr = ReductionKindAttr.get(ctx, .{ .value = self }) catch unreachable;
        return attr.attribute();
    }

    fn forInput(self: ReductionKind, input: *const mlir.Value, unsigned_integer: bool) ReductionKind {
        if (self != .max and self != .min) return self;
        const element_type = input.type_().isA(mlir.ShapedType).?.elementType();
        if (element_type.isFloat()) return if (self == .max) .maxf else .minf;
        // Match Mosaic's serde upgrade: scans used unsigned integer extrema,
        // while all-reduce used signed integer extrema.
        return if (unsigned_integer)
            (if (self == .max) .maxui else .minui)
        else
            (if (self == .max) .maxsi else .minsi);
    }
};

/// `#tpu.reduction_kind<...>`.
pub const ReductionKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsATpuReductionKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "reduction_kind";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: ReductionKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirTpuReductionKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) ReductionKind {
        return @enumFromInt(c.mlirTpuReductionKindAttrGetValue(self.ptr()));
    }
};

/// Matmul precision setting of `tpu.matmul`.
pub const ContractPrecision = enum(u32) {
    bf16 = 0,
    fp32 = 1,
    bf16x3 = 2,

    pub fn attribute(self: ContractPrecision, ctx: *mlir.Context) *const mlir.Attribute {
        const attr = ContractPrecisionAttr.get(ctx, .{ .value = self }) catch unreachable;
        return attr.attribute();
    }
};

/// `#tpu.contract_precision<...>`.
pub const ContractPrecisionAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsATpuContractPrecision;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "contract_precision";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: ContractPrecision,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirTpuContractPrecisionAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) ContractPrecision {
        return @enumFromInt(c.mlirTpuContractPrecisionAttrGetValue(self.ptr()));
    }
};

/// Rounding mode of `tpu.fptosi` / `tpu.fptoui` / etc.
pub const RoundingMode = enum(u32) {
    towards_zero = 0,
    to_nearest_even = 1,

    pub fn attribute(self: RoundingMode, ctx: *mlir.Context) *const mlir.Attribute {
        const attr = RoundingModeAttr.get(ctx, .{ .value = self }) catch unreachable;
        return attr.attribute();
    }
};

/// `#tpu.rounding_mode<...>`.
pub const RoundingModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsATpuRoundingMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "rounding_mode";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: RoundingMode,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirTpuRoundingModeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) RoundingMode {
        return @enumFromInt(c.mlirTpuRoundingModeAttrGetValue(self.ptr()));
    }
};

/// Target core of a kernel, remote DMAs and `tpu.sem_signal`.
pub const CoreType = enum(u32) {
    tc = 0,
    sc_scalar_subcore = 1,
    sc_vector_subcore = 2,

    pub fn attribute(self: CoreType, ctx: *mlir.Context) *const mlir.Attribute {
        const attr = CoreTypeAttr.get(ctx, .{ .value = self }) catch unreachable;
        return attr.attribute();
    }
};

/// `#tpu.core_type<...>`.
pub const CoreTypeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsATpuCoreType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "core_type";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: CoreType,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirTpuCoreTypeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) CoreType {
        return @enumFromInt(c.mlirTpuCoreTypeAttrGetValue(self.ptr()));
    }
};

/// Grid dimension semantics annotation.
pub const DimensionSemantics = enum(u32) {
    parallel = 0,
    arbitrary = 1,
    core_parallel = 2,
    subcore_parallel = 3,

    pub fn attribute(self: DimensionSemantics, ctx: *mlir.Context) *const mlir.Attribute {
        const attr = DimensionSemanticsAttr.get(ctx, .{ .value = self }) catch unreachable;
        return attr.attribute();
    }
};

/// `#tpu.dimension_semantics<...>`.
pub const DimensionSemanticsAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsATpuDimensionSemantics;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "dimension_semantics";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: DimensionSemantics,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirTpuDimensionSemanticsAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) DimensionSemantics {
        return @enumFromInt(c.mlirTpuDimensionSemanticsAttrGetValue(self.ptr()));
    }
};

/// Buffering mode of a `window_params` entry.
pub const PipelineMode = enum(u32) {
    synchronous = 1,
    double_buffered = 2,

    pub fn attribute(self: PipelineMode, ctx: *mlir.Context) *const mlir.Attribute {
        const attr = PipelineModeAttr.get(ctx, .{ .value = self }) catch unreachable;
        return attr.attribute();
    }
};

/// `#tpu.pipeline_mode<...>`.
pub const PipelineModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsATpuPipelineMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "pipeline_mode";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: PipelineMode,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirTpuPipelineModeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) PipelineMode {
        return @enumFromInt(c.mlirTpuPipelineModeAttrGetValue(self.ptr()));
    }
};

/// Block revisit mode of a `window_params` entry.
pub const RevisitMode = enum(u32) {
    immediate = 0,
    any = 1,

    pub fn attribute(self: RevisitMode, ctx: *mlir.Context) *const mlir.Attribute {
        const attr = RevisitModeAttr.get(ctx, .{ .value = self }) catch unreachable;
        return attr.attribute();
    }
};

/// `#tpu.revisit_mode<...>`.
pub const RevisitModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsATpuRevisitMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "revisit_mode";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: RevisitMode,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirTpuRevisitModeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) RevisitMode {
        return @enumFromInt(c.mlirTpuRevisitModeAttrGetValue(self.ptr()));
    }
};

/// Memory space of a memref, `#tpu.memory_space<...>`.
pub const MemorySpace = enum(u32) {
    any = 0xffff_ffff,
    vmem = 0,
    smem = 1,
    hbm = 2,
    cmem = 3,
    semaphore_mem = 4,
    vmem_shared = 5,
    host = 6,

    pub fn attribute(self: MemorySpace, ctx: *mlir.Context) *const mlir.Attribute {
        const attr = MemorySpaceAttr.get(ctx, .{ .value = self }) catch unreachable;
        return attr.attribute();
    }
};

/// `#tpu.memory_space<value[, core_type]>` — passed as the `memory_space` of a memref's type.
pub const MemorySpaceAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsATpuMemorySpace;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "memory_space";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MemorySpace,
        coreType: ?CoreType = null,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const core_type: i64 = if (args.coreType) |ct| @intFromEnum(ct) else -1;
        const result = c.mlirTpuMemorySpaceAttrGet(ctx.ptr(), @intFromEnum(args.value), core_type);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) MemorySpace {
        return @enumFromInt(c.mlirTpuMemorySpaceAttrGetValue(self.ptr()));
    }
    pub fn getCoreType(self: *const Self) ?CoreType {
        const core_type = c.mlirTpuMemorySpaceAttrGetCoreType(self.ptr());
        return if (core_type < 0) null else @enumFromInt(core_type);
    }
};

/// `#tpu.element_window<[pad_low], [pad_high]>` — per-arg window padding.
pub const ElementWindowAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsATpuElementWindow;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "element_window";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        padLow: []const i64,
        padHigh: []const i64,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirTpuElementWindowAttrGet(
            ctx.ptr(),
            @intCast(args.padLow.len),
            args.padLow.ptr,
            @intCast(args.padHigh.len),
            args.padHigh.ptr,
        );
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getNumPadLow(self: *const Self) usize {
        return @intCast(c.mlirTpuElementWindowAttrGetNumPadLow(self.ptr()));
    }
    pub fn getPadLow(self: *const Self, pos: usize) i64 {
        return c.mlirTpuElementWindowAttrGetPadLow(self.ptr(), @intCast(pos));
    }
    pub fn getNumPadHigh(self: *const Self) usize {
        return @intCast(c.mlirTpuElementWindowAttrGetNumPadHigh(self.ptr()));
    }
    pub fn getPadHigh(self: *const Self, pos: usize) i64 {
        return c.mlirTpuElementWindowAttrGetPadHigh(self.ptr(), @intCast(pos));
    }
};

/// `#tpu.element_window<[pad_low], [pad_high]>` — per-arg window padding.
pub fn elementWindowAttribute(
    ctx: *mlir.Context,
    pad_low: []const i64,
    pad_high: []const i64,
) *const mlir.Attribute {
    const attr = ElementWindowAttr.get(ctx, .{ .padLow = pad_low, .padHigh = pad_high }) catch unreachable;
    return attr.attribute();
}

/// `#tpu.dot_dimension_numbers<...>` — the dimension mapping of `tpu.matmul`.
pub const DotDimensionNumbersAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsATpuDotDimensionNumbers;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "dot_dimension_numbers";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        lhsContractingDims: []const i64,
        rhsContractingDims: []const i64,
        lhsNonContractingDims: []const i64,
        /// Empty when rhs is a 1-D vector.
        rhsNonContractingDims: []const i64 = &.{},
        /// Flattened (operand, dim) pairs: operand 0 is lhs, 1 is rhs.
        outputDimOrder: []const i64,
        lhsBatchDims: []const i64 = &.{},
        rhsBatchDims: []const i64 = &.{},
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirTpuDotDimensionNumbersAttrGet(
            ctx.ptr(),
            @intCast(args.lhsContractingDims.len),
            args.lhsContractingDims.ptr,
            @intCast(args.rhsContractingDims.len),
            args.rhsContractingDims.ptr,
            @intCast(args.lhsNonContractingDims.len),
            args.lhsNonContractingDims.ptr,
            @intCast(args.rhsNonContractingDims.len),
            args.rhsNonContractingDims.ptr,
            @intCast(args.outputDimOrder.len),
            args.outputDimOrder.ptr,
            @intCast(args.lhsBatchDims.len),
            args.lhsBatchDims.ptr,
            @intCast(args.rhsBatchDims.len),
            args.rhsBatchDims.ptr,
        );
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getNumLhsContractingDims(self: *const Self) usize {
        return @intCast(c.mlirTpuDotDimensionNumbersAttrGetNumLhsContractingDims(self.ptr()));
    }
    pub fn getLhsContractingDims(self: *const Self, pos: usize) i64 {
        return c.mlirTpuDotDimensionNumbersAttrGetLhsContractingDims(self.ptr(), @intCast(pos));
    }
    pub fn getNumRhsContractingDims(self: *const Self) usize {
        return @intCast(c.mlirTpuDotDimensionNumbersAttrGetNumRhsContractingDims(self.ptr()));
    }
    pub fn getRhsContractingDims(self: *const Self, pos: usize) i64 {
        return c.mlirTpuDotDimensionNumbersAttrGetRhsContractingDims(self.ptr(), @intCast(pos));
    }
    pub fn getNumLhsNonContractingDims(self: *const Self) usize {
        return @intCast(c.mlirTpuDotDimensionNumbersAttrGetNumLhsNonContractingDims(self.ptr()));
    }
    pub fn getLhsNonContractingDims(self: *const Self, pos: usize) i64 {
        return c.mlirTpuDotDimensionNumbersAttrGetLhsNonContractingDims(self.ptr(), @intCast(pos));
    }
    pub fn getNumRhsNonContractingDims(self: *const Self) usize {
        return @intCast(c.mlirTpuDotDimensionNumbersAttrGetNumRhsNonContractingDims(self.ptr()));
    }
    pub fn getRhsNonContractingDims(self: *const Self, pos: usize) i64 {
        return c.mlirTpuDotDimensionNumbersAttrGetRhsNonContractingDims(self.ptr(), @intCast(pos));
    }
    pub fn getNumOutputDimOrder(self: *const Self) usize {
        return @intCast(c.mlirTpuDotDimensionNumbersAttrGetNumOutputDimOrder(self.ptr()));
    }
    pub fn getOutputDimOrder(self: *const Self, pos: usize) i64 {
        return c.mlirTpuDotDimensionNumbersAttrGetOutputDimOrder(self.ptr(), @intCast(pos));
    }
    pub fn getNumLhsBatchDims(self: *const Self) usize {
        return @intCast(c.mlirTpuDotDimensionNumbersAttrGetNumLhsBatchDims(self.ptr()));
    }
    pub fn getLhsBatchDims(self: *const Self, pos: usize) i64 {
        return c.mlirTpuDotDimensionNumbersAttrGetLhsBatchDims(self.ptr(), @intCast(pos));
    }
    pub fn getNumRhsBatchDims(self: *const Self) usize {
        return @intCast(c.mlirTpuDotDimensionNumbersAttrGetNumRhsBatchDims(self.ptr()));
    }
    pub fn getRhsBatchDims(self: *const Self, pos: usize) i64 {
        return c.mlirTpuDotDimensionNumbersAttrGetRhsBatchDims(self.ptr(), @intCast(pos));
    }
};

// =============================================================================
// Reductions / scan / sort
// =============================================================================

pub fn all_reduce(
    ctx: *mlir.Context,
    input: *const mlir.Value,
    dim: i64,
    kind: ReductionKind,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.all_reduce", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "dim", .int(ctx, .i64, dim)),
            .named(ctx, "kind", kind.forInput(input, false).attribute(ctx)),
        },
        .location = location,
    });
}

pub fn reduce_index(
    ctx: *mlir.Context,
    input: *const mlir.Value,
    axis: i32,
    kind: ReductionKind,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.reduce_index", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "axis", .int(ctx, .i32, axis)),
            .named(ctx, "kind", kind.attribute(ctx)),
        },
        .location = location,
    });
}

/// Scan the last dimension; an optional rank-one mask spans that dimension.
pub fn scan(
    ctx: *mlir.Context,
    input: *const mlir.Value,
    kind: ReductionKind,
    mask: ?*const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 2) = .empty;
    operands_buf.appendAssumeCapacity(input);
    if (mask) |m| operands_buf.appendAssumeCapacity(m);

    return mlir.Operation.make(ctx, "tpu.scan", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "kind", kind.forInput(input, true).attribute(ctx)),
            .named(ctx, "dimension", .int(ctx, .i64, @as(i64, @intCast(input.type_().isA(mlir.ShapedType).?.rank())) - 1)),
        },
        .location = location,
    });
}

pub const SortOpts = struct {
    descending: bool = false,
};

pub fn sort(
    ctx: *mlir.Context,
    keys: *const mlir.Value,
    values: *const mlir.Value,
    mask: ?*const mlir.Value,
    opts: SortOpts,
    output_mask_type: *const mlir.Type,
    sorted_keys_type: *const mlir.Type,
    sorted_values_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 3) = .empty;
    operands_buf.appendSliceAssumeCapacity(&.{ keys, values });
    if (mask) |m| operands_buf.appendAssumeCapacity(m);

    return mlir.Operation.make(ctx, "tpu.sort", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{ output_mask_type, sorted_keys_type, sorted_values_type } },
        .attributes = &.{
            .named(ctx, "descending", .boolean(ctx, opts.descending)),
        },
        .location = location,
    });
}

// =============================================================================
// Memory — load / store / vector_load / vector_store / strided_*
// =============================================================================

pub const LoadOpts = struct {
    sublane_mask: []const bool,
    sublane_stride: i32 = 1,
};

/// tpu.load — sublane-aware load into a vreg.
pub fn load(
    ctx: *mlir.Context,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    opts: LoadOpts,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 16) = .empty;
    operands_buf.appendAssumeCapacity(base);
    operands_buf.appendSliceAssumeCapacity(indices);

    return mlir.Operation.make(ctx, "tpu.load", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "sublane_mask", denseBoolArrayAttribute(ctx, opts.sublane_mask)),
            .named(ctx, "sublane_stride", .int(ctx, .i32, opts.sublane_stride)),
        },
        .location = location,
    });
}

pub const StoreOpts = struct {
    sublane_mask: []const bool,
    sublane_stride: i32 = 1,
    add: bool = false,
};

/// tpu.store — sublane-aware store from a vreg. Optional elementwise mask.
pub fn store(
    ctx: *mlir.Context,
    value: *const mlir.Value,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    mask: ?*const mlir.Value,
    opts: StoreOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 18) = .empty;
    operands_buf.appendSliceAssumeCapacity(&.{ value, base });
    operands_buf.appendSliceAssumeCapacity(indices);
    const mask_len: i32 = if (mask) |m| blk: {
        operands_buf.appendAssumeCapacity(m);
        break :blk 1;
    } else 0;

    const seg_sizes = [4]i32{ 1, 1, @intCast(indices.len), mask_len };
    return mlir.Operation.make(ctx, "tpu.store", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &seg_sizes)),
            .named(ctx, "sublane_mask", denseBoolArrayAttribute(ctx, opts.sublane_mask)),
            .named(ctx, "sublane_stride", .int(ctx, .i32, opts.sublane_stride)),
            .named(ctx, "add", .boolean(ctx, opts.add)),
        },
        .location = location,
    });
}

pub const VectorLoadOpts = struct {
    /// Per-dim load stride. Pass `&.{}` for unit-strided.
    strides: []const i32 = &.{},
};

/// tpu.vector_load — multi-dim strided load with optional elementwise mask.
pub fn vector_load(
    ctx: *mlir.Context,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    mask: ?*const mlir.Value,
    opts: VectorLoadOpts,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 17) = .empty;
    operands_buf.appendAssumeCapacity(base);
    operands_buf.appendSliceAssumeCapacity(indices);
    const mask_len: i32 = if (mask) |m| blk: {
        operands_buf.appendAssumeCapacity(m);
        break :blk 1;
    } else 0;

    const seg_sizes = [3]i32{ 1, @intCast(indices.len), mask_len };
    return mlir.Operation.make(ctx, "tpu.vector_load", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &seg_sizes)),
            .named(ctx, "strides", .denseArray(ctx, .i32, opts.strides)),
        },
        .location = location,
    });
}

pub const VectorStoreOpts = struct {
    strides: []const i32 = &.{},
    add: bool = false,
};

pub fn vector_store(
    ctx: *mlir.Context,
    value: *const mlir.Value,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    mask: ?*const mlir.Value,
    opts: VectorStoreOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 18) = .empty;
    operands_buf.appendSliceAssumeCapacity(&.{ value, base });
    operands_buf.appendSliceAssumeCapacity(indices);
    const mask_len: i32 = if (mask) |m| blk: {
        operands_buf.appendAssumeCapacity(m);
        break :blk 1;
    } else 0;

    const seg_sizes = [4]i32{ 1, 1, @intCast(indices.len), mask_len };
    return mlir.Operation.make(ctx, "tpu.vector_store", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &seg_sizes)),
            .named(ctx, "strides", .denseArray(ctx, .i32, opts.strides)),
            .named(ctx, "add", .boolean(ctx, opts.add)),
        },
        .location = location,
    });
}

pub fn strided_load(
    ctx: *mlir.Context,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    strides: []const i32,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 16) = .empty;
    operands_buf.appendAssumeCapacity(base);
    operands_buf.appendSliceAssumeCapacity(indices);

    return mlir.Operation.make(ctx, "tpu.strided_load", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "strides", .denseArray(ctx, .i32, strides)),
        },
        .location = location,
    });
}

pub fn strided_store(
    ctx: *mlir.Context,
    value: *const mlir.Value,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    strides: []const i32,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 17) = .empty;
    operands_buf.appendSliceAssumeCapacity(&.{ value, base });
    operands_buf.appendSliceAssumeCapacity(indices);

    return mlir.Operation.make(ctx, "tpu.strided_store", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .attributes = &.{
            .named(ctx, "strides", .denseArray(ctx, .i32, strides)),
        },
        .location = location,
    });
}

// =============================================================================
// Compute — matmul, iota, reciprocal, casts
// =============================================================================

pub const MatmulOpts = struct {
    /// Deprecated when `dimension_numbers` is provided.
    transpose_lhs: bool = false,
    transpose_rhs: bool = false,
    transpose_lhs_hint: bool = false,
    /// Optional precision; pass null to omit.
    precision: ?ContractPrecision = null,
    /// Optional `#tpu.dot_dimension_numbers<...>` attribute. When omitted the
    /// canonicalizer derives one. Build with `DotDimensionNumbersAttr` or
    /// `dotDimensionNumbers`.
    dimension_numbers: ?*const mlir.Attribute = null,
};

pub fn matmul(
    ctx: *mlir.Context,
    lhs: *const mlir.Value,
    rhs: *const mlir.Value,
    acc: *const mlir.Value,
    opts: MatmulOpts,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    attrs.appendSliceAssumeCapacity(&.{
        .named(ctx, "transpose_lhs", .boolean(ctx, opts.transpose_lhs)),
        .named(ctx, "transpose_rhs", .boolean(ctx, opts.transpose_rhs)),
        .named(ctx, "transpose_lhs_hint", .boolean(ctx, opts.transpose_lhs_hint)),
    });
    if (opts.precision) |p| {
        attrs.appendAssumeCapacity(.named(ctx, "precision", p.attribute(ctx)));
    }
    if (opts.dimension_numbers) |dn| {
        attrs.appendAssumeCapacity(.named(ctx, "dimension_numbers", dn));
    }

    return mlir.Operation.make(ctx, "tpu.matmul", .{
        .operands = .{ .flat = &.{ lhs, rhs, acc } },
        .results = .{ .flat = &.{result_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

pub fn iota(
    ctx: *mlir.Context,
    dimensions: ?[]const i32,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (dimensions) |d| {
        attrs.appendAssumeCapacity(.named(ctx, "dimensions", .denseArray(ctx, .i32, d)));
    }
    return mlir.Operation.make(ctx, "tpu.iota", .{
        .results = .{ .flat = &.{result_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

pub const ReciprocalOpts = struct {
    approx: bool = false,
    full_range: bool = false,
};

pub fn reciprocal(
    ctx: *mlir.Context,
    input: *const mlir.Value,
    opts: ReciprocalOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.reciprocal", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{input.type_()} },
        .attributes = &.{
            .named(ctx, "approx", .boolean(ctx, opts.approx)),
            .named(ctx, "full_range", .boolean(ctx, opts.full_range)),
        },
        .location = location,
    });
}

/// Generic 1-operand → 1-result builder for the various TPU casts.
fn castOp(comptime op_name: []const u8) type {
    return struct {
        pub fn call(
            ctx: *mlir.Context,
            src: *const mlir.Value,
            result_type: *const mlir.Type,
            location: *const mlir.Location,
        ) *mlir.Operation {
            return mlir.Operation.make(ctx, op_name, .{
                .operands = .{ .flat = &.{src} },
                .results = .{ .flat = &.{result_type} },
                .location = location,
            });
        }
    };
}

pub const fptosi = castOp("tpu.fptosi").call;
pub const fptoui = castOp("tpu.fptoui").call;
pub const sitofp = castOp("tpu.sitofp").call;
pub const uitofp = castOp("tpu.uitofp").call;
pub const extf = castOp("tpu.extf").call;
pub const truncf = castOp("tpu.truncf").call;
pub const bitcast = castOp("tpu.bitcast").call;
pub const bitcast_vreg = castOp("tpu.bitcast_vreg").call;
pub const mask_cast = castOp("tpu.mask_cast").call;

// =============================================================================
// Shape — reshape / repeat / concatenate / transpose / broadcast_in_sublanes
// =============================================================================

pub fn reshape(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.reshape", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

pub fn repeat(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    dimension: i32,
    times: i32,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.repeat", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "dimension", .int(ctx, .i32, dimension)),
            .named(ctx, "times", .int(ctx, .i32, times)),
        },
        .location = location,
    });
}

pub fn concatenate(
    ctx: *mlir.Context,
    sources: []const *const mlir.Value,
    dimension: i32,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.concatenate", .{
        .operands = .{ .flat = sources },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "dimension", .int(ctx, .i32, dimension)),
        },
        .location = location,
    });
}

pub fn transpose(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    permutation: []const i64,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.transpose", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "permutation", .denseArray(ctx, .i64, permutation)),
        },
        .location = location,
    });
}

pub fn broadcast_in_sublanes(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    lane: i32,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.broadcast_in_sublanes", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "lane", .int(ctx, .i32, lane)),
        },
        .location = location,
    });
}

pub const DynamicRotateOpts = struct {
    dimension: i32,
    stride: ?i32 = null,
    stride_dimension: ?i32 = null,
};

pub fn dynamic_rotate(
    ctx: *mlir.Context,
    value: *const mlir.Value,
    amount: *const mlir.Value,
    opts: DynamicRotateOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attrs.appendSliceAssumeCapacity(&.{
        .named(ctx, "dimension", .int(ctx, .si32, opts.dimension)),
    });
    if (opts.stride) |s| attrs.appendAssumeCapacity(.named(ctx, "stride", .int(ctx, .si32, s)));
    if (opts.stride_dimension) |sd| attrs.appendAssumeCapacity(.named(ctx, "stride_dimension", .int(ctx, .si32, sd)));

    return mlir.Operation.make(ctx, "tpu.dynamic_rotate", .{
        .operands = .{ .flat = &.{ value, amount } },
        .results = .{ .flat = &.{value.type_()} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

// =============================================================================
// Memref-shaped ops — slice / squeeze / reshape / bitcast / reinterpret_cast
// =============================================================================

pub const MemRefSliceArgs = struct {
    /// `base_idx` count must equal source memref rank.
    base_idx: []const *const mlir.Value,
    /// Optional dynamic sizes; left empty when target shape is fully static.
    dynamic_sizes: []const *const mlir.Value = &.{},
};

pub fn memref_slice(
    ctx: *mlir.Context,
    mem_ref: *const mlir.Value,
    args: MemRefSliceArgs,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 32) = .empty;
    operands_buf.appendAssumeCapacity(mem_ref);
    operands_buf.appendSliceAssumeCapacity(args.base_idx);
    operands_buf.appendSliceAssumeCapacity(args.dynamic_sizes);

    const seg_sizes = [3]i32{ 1, @intCast(args.base_idx.len), @intCast(args.dynamic_sizes.len) };
    return mlir.Operation.make(ctx, "tpu.memref_slice", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &seg_sizes)),
        },
        .location = location,
    });
}

pub fn memref_squeeze(
    ctx: *mlir.Context,
    mem_ref: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.memref_squeeze", .{
        .operands = .{ .flat = &.{mem_ref} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

pub fn memref_reshape(
    ctx: *mlir.Context,
    mem_ref: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.memref_reshape", .{
        .operands = .{ .flat = &.{mem_ref} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

pub fn memref_bitcast(
    ctx: *mlir.Context,
    mem_ref: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.memref_bitcast", .{
        .operands = .{ .flat = &.{mem_ref} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

pub fn reinterpret_cast(
    ctx: *mlir.Context,
    mem_ref: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.reinterpret_cast", .{
        .operands = .{ .flat = &.{mem_ref} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{.named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &.{ 1, 0, 0, 0 }))},
        .location = location,
    });
}

pub fn assume_layout(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.assume_layout", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

pub fn erase_memref_layout(
    ctx: *mlir.Context,
    mem_ref: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.erase_memref_layout", .{
        .operands = .{ .flat = &.{mem_ref} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// =============================================================================
// Mask building — create_mask / create_subelement_mask / mask_cast
// =============================================================================

/// `tpu.create_mask` — two equal-length operand groups (low, high) per dim.
pub fn create_mask(
    ctx: *mlir.Context,
    low_bounds: []const *const mlir.Value,
    high_bounds: []const *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    std.debug.assert(low_bounds.len == high_bounds.len);
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 32) = .empty;
    operands_buf.appendSliceAssumeCapacity(low_bounds);
    operands_buf.appendSliceAssumeCapacity(high_bounds);

    return mlir.Operation.make(ctx, "tpu.create_mask", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// =============================================================================
// Semaphores / DMA / barriers
// =============================================================================

pub fn sem_alloc(
    ctx: *mlir.Context,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.sem_alloc", .{
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

pub fn sem_barrier(
    ctx: *mlir.Context,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.sem_barrier", .{
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

pub fn sem_read(
    ctx: *mlir.Context,
    semaphore: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.sem_read", .{
        .operands = .{ .flat = &.{semaphore} },
        .results = .{ .flat = &.{.int(ctx, .i32)} },
        .location = location,
    });
}

pub fn sem_wait(
    ctx: *mlir.Context,
    semaphore: *const mlir.Value,
    amount: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.sem_wait", .{
        .operands = .{ .flat = &.{ semaphore, amount } },
        .location = location,
    });
}

pub const SemSignalOpts = struct {
    device_id: ?*const mlir.Value = null,
    core_id: ?*const mlir.Value = null,
    subcore_id: ?*const mlir.Value = null,
};

pub fn sem_signal(
    ctx: *mlir.Context,
    semaphore: *const mlir.Value,
    amount: *const mlir.Value,
    opts: SemSignalOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 5) = .empty;
    operands_buf.appendSliceAssumeCapacity(&.{ semaphore, amount });
    const dev_len: i32 = if (opts.device_id) |d| blk: {
        operands_buf.appendAssumeCapacity(d);
        break :blk 1;
    } else 0;
    const core_len: i32 = if (opts.core_id) |c_| blk: {
        operands_buf.appendAssumeCapacity(c_);
        break :blk 1;
    } else 0;

    const subcore_len: i32 = if (opts.subcore_id) |subcore| blk: {
        operands_buf.appendAssumeCapacity(subcore);
        break :blk 1;
    } else 0;

    const seg_sizes = [5]i32{ 1, 1, dev_len, core_len, subcore_len };
    return mlir.Operation.make(ctx, "tpu.sem_signal", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &seg_sizes)),
        },
        .location = location,
    });
}

pub fn barrier(
    ctx: *mlir.Context,
    barrier_id: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.barrier", .{
        .operands = .{ .flat = &.{barrier_id} },
        .location = location,
    });
}

pub const EnqueueDmaOpts = struct {
    source_semaphore: ?*const mlir.Value = null,
    device_id: ?*const mlir.Value = null,
    core_id: ?*const mlir.Value = null,
    subcore_id: ?*const mlir.Value = null,
    priority: i32 = 0,
    strict_ordering: bool = false,
};

pub fn enqueue_dma(
    ctx: *mlir.Context,
    source: *const mlir.Value,
    target: *const mlir.Value,
    target_semaphore: *const mlir.Value,
    opts: EnqueueDmaOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 7) = .empty;
    operands_buf.appendAssumeCapacity(source);
    const src_sem_len: i32 = if (opts.source_semaphore) |s| blk: {
        operands_buf.appendAssumeCapacity(s);
        break :blk 1;
    } else 0;
    operands_buf.appendAssumeCapacity(target);
    operands_buf.appendAssumeCapacity(target_semaphore);
    const dev_len: i32 = if (opts.device_id) |d| blk: {
        operands_buf.appendAssumeCapacity(d);
        break :blk 1;
    } else 0;
    const core_len: i32 = if (opts.core_id) |c_| blk: {
        operands_buf.appendAssumeCapacity(c_);
        break :blk 1;
    } else 0;

    // Source / source_semaphore? / target / target_semaphore / device_id? / core_id? / subcore_id?
    const subcore_len: i32 = if (opts.subcore_id) |subcore| blk: {
        operands_buf.appendAssumeCapacity(subcore);
        break :blk 1;
    } else 0;

    const seg_sizes = [7]i32{ 1, src_sem_len, 1, 1, dev_len, core_len, subcore_len };
    return mlir.Operation.make(ctx, "tpu.enqueue_dma", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &seg_sizes)),
            .named(ctx, "priority", .int(ctx, .i32, opts.priority)),
            .named(ctx, "strict_ordering", .boolean(ctx, opts.strict_ordering)),
        },
        .location = location,
    });
}

pub const WaitDma2Opts = struct {
    device_id: ?*const mlir.Value = null,
    core_id: ?*const mlir.Value = null,
    strict_ordering: bool = false,
};

pub fn wait_dma2(
    ctx: *mlir.Context,
    semaphore: *const mlir.Value,
    src: *const mlir.Value,
    dst: *const mlir.Value,
    opts: WaitDma2Opts,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 5) = .empty;
    operands_buf.appendSliceAssumeCapacity(&.{ semaphore, src, dst });
    const dev_len: i32 = if (opts.device_id) |d| blk: {
        operands_buf.appendAssumeCapacity(d);
        break :blk 1;
    } else 0;
    const core_len: i32 = if (opts.core_id) |c_| blk: {
        operands_buf.appendAssumeCapacity(c_);
        break :blk 1;
    } else 0;

    const seg_sizes = [5]i32{ 1, 1, 1, dev_len, core_len };
    return mlir.Operation.make(ctx, "tpu.wait_dma2", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &seg_sizes)),
            .named(ctx, "strict_ordering", .boolean(ctx, opts.strict_ordering)),
        },
        .location = location,
    });
}

// =============================================================================
// Region-bearing — region / trace / yield
// =============================================================================

/// tpu.region — single-block region with an implicit `tpu.yield` terminator.
pub fn region(
    ctx: *mlir.Context,
    body: *mlir.Block,
    result_types: []const *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.region", .{
        .results = .{ .flat = result_types },
        .blocks = &.{body},
        .verify = false,
        .location = location,
    });
}

pub fn trace(
    ctx: *mlir.Context,
    body: *mlir.Block,
    result_types: []const *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.trace", .{
        .results = .{ .flat = result_types },
        .blocks = &.{body},
        .verify = false,
        .location = location,
    });
}

pub fn yield(
    ctx: *mlir.Context,
    values: []const *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.yield", .{
        .operands = .{ .flat = values },
        .verify = false,
        .location = location,
    });
}

pub fn trace_start(
    ctx: *mlir.Context,
    level: i32,
    message: []const u8,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.trace_start", .{
        .attributes = &.{
            .named(ctx, "level", .int(ctx, .i32, level)),
            .named(ctx, "message", .string(ctx, message)),
        },
        .verify = false,
        .location = location,
    });
}

pub fn trace_stop(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.trace_stop", .{
        .verify = false,
        .location = location,
    });
}

/// Build a `#tpu.dot_dimension_numbers<...>` attribute. Mirrors
/// `tpu_ops.td:TPU_DotDimensionNumbersAttr`. Pass empty slices for the
/// fields you don't use (most dot operations leave batch dims empty).
pub fn dotDimensionNumbers(
    ctx: *mlir.Context,
    lhs_contracting: []const i64,
    rhs_contracting: []const i64,
    lhs_non_contracting: []const i64,
    rhs_non_contracting: []const i64,
    output_dim_order: []const i64,
    lhs_batch: []const i64,
    rhs_batch: []const i64,
) *const mlir.Attribute {
    const attr = DotDimensionNumbersAttr.get(ctx, .{
        .lhsContractingDims = lhs_contracting,
        .rhsContractingDims = rhs_contracting,
        .lhsNonContractingDims = lhs_non_contracting,
        .rhsNonContractingDims = rhs_non_contracting,
        .outputDimOrder = output_dim_order,
        .lhsBatchDims = lhs_batch,
        .rhsBatchDims = rhs_batch,
    }) catch unreachable;
    return attr.attribute();
}

// =============================================================================
// Misc — device_id / delay / assume_multiple
// =============================================================================

pub fn device_id(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.device_id", .{
        .results = .{ .flat = &.{.int(ctx, .i32)} },
        .location = location,
    });
}

pub fn delay(
    ctx: *mlir.Context,
    cycles: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.delay", .{
        .operands = .{ .flat = &.{cycles} },
        .location = location,
    });
}

pub fn assume_multiple(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    multiple: i32,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.assume_multiple", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{src.type_()} },
        .attributes = &.{
            .named(ctx, "multiple", .int(ctx, .i32, multiple)),
        },
        .location = location,
    });
}

// =============================================================================
// Internal helpers
// =============================================================================

/// `tpu.DenseBoolArrayAttr` stores one C `int` (i32) per bool, see `mlirDenseBoolArrayGet`
fn denseBoolArrayAttribute(ctx: *mlir.Context, values: []const bool) *const mlir.Attribute {
    var ints: stdx.BoundedArray(i32, 64) = .empty;
    for (values) |b| ints.appendAssumeCapacity(@intFromBool(b));
    return .denseArray(ctx, .bool, ints.constSlice());
}

test {
    std.testing.refAllDecls(@This());
}

fn testContext() !*mlir.Context {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    registry.registerDialect("tpu");
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();
    return ctx;
}

fn expectSegments(op: *const mlir.Operation, expected: []const i32) !void {
    const segments = op.attributeByName("operandSegmentSizes").?.isA(mlir.DenseArrayAttribute(.i32)).?;
    try std.testing.expectEqual(expected.len, segments.numElements());
    for (expected, 0..) |size, i| try std.testing.expectEqual(size, segments.element(i));
}

test "DMA and semaphore wrappers satisfy current operand groups" {
    const ctx = try testContext();
    defer ctx.deinit();
    const loc = mlir.Location.unknown(ctx);
    const args = mlir.Block.init(&.{}, &.{});
    defer args.deinit();
    const source = args.addArgument(try mlir.Type.parse(ctx, "memref<8x128xbf16, #tpu.memory_space<hbm>>"), loc);
    const target = args.addArgument(try mlir.Type.parse(ctx, "memref<8x128xbf16, #tpu.memory_space<vmem>>"), loc);
    const dma_sem = args.addArgument(try mlir.Type.parse(ctx, "memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>"), loc);
    const sem = args.addArgument(try mlir.Type.parse(ctx, "memref<!tpu.semaphore, #tpu.memory_space<semaphore_mem>>"), loc);
    const amount = args.addArgument(.int(ctx, .i32), loc);

    const dma = enqueue_dma(ctx, source, target, dma_sem, .{}, loc);
    defer dma.deinit();
    try expectSegments(dma, &.{ 1, 0, 1, 1, 0, 0, 0 });
    try std.testing.expect(dma.verify());
    const signal = sem_signal(ctx, sem, amount, .{}, loc);
    defer signal.deinit();
    try expectSegments(signal, &.{ 1, 1, 0, 0, 0 });
    try std.testing.expect(signal.verify());

    const sc_sem = args.addArgument(try mlir.Type.parse(ctx, "memref<!tpu.semaphore, #tpu.memory_space<semaphore_mem, sc_vector_subcore>>"), loc);
    const sc_dma_sem = args.addArgument(try mlir.Type.parse(ctx, "memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem, sc_vector_subcore>>"), loc);
    const core = args.addArgument(.int(ctx, .i32), loc);
    const subcore = args.addArgument(.int(ctx, .i32), loc);
    const remote_signal = sem_signal(ctx, sc_sem, amount, .{ .core_id = core, .subcore_id = subcore }, loc);
    defer remote_signal.deinit();
    try expectSegments(remote_signal, &.{ 1, 1, 0, 1, 1 });
    try std.testing.expect(remote_signal.operand(3).eql(subcore));
    const remote_dma = enqueue_dma(ctx, source, target, sc_dma_sem, .{
        .source_semaphore = dma_sem,
        .core_id = core,
        .subcore_id = subcore,
    }, loc);
    defer remote_dma.deinit();
    try expectSegments(remote_dma, &.{ 1, 1, 1, 1, 0, 1, 1 });
    try std.testing.expect(remote_dma.operand(5).eql(subcore));
}

test "matmul and reinterpret cast satisfy current required properties" {
    const ctx = try testContext();
    defer ctx.deinit();
    const loc = mlir.Location.unknown(ctx);
    const args = mlir.Block.init(&.{}, &.{});
    defer args.deinit();
    const vector_type = mlir.Type.vector(&.{ 8, 8 }, .float(ctx, .f32));
    const lhs = args.addArgument(vector_type, loc);
    const rhs = args.addArgument(vector_type, loc);
    const acc = args.addArgument(vector_type, loc);
    const product = matmul(ctx, lhs, rhs, acc, .{}, vector_type, loc);
    defer product.deinit();
    try std.testing.expect(product.verify());
    try std.testing.expect(!product.attributeByName("transpose_lhs_hint").?.isA(mlir.BoolAttribute).?.value());

    const ref_type = try mlir.Type.parse(ctx, "memref<8x8xf32, #tpu.memory_space<vmem>>");
    const ref = args.addArgument(ref_type, loc);
    const cast = reinterpret_cast(ctx, ref, ref_type, loc);
    defer cast.deinit();
    try expectSegments(cast, &.{ 1, 0, 0, 0 });
    try std.testing.expect(cast.verify());
}

test "legacy reduction extrema and scan axis preserve their semantics" {
    const ctx = try testContext();
    defer ctx.deinit();
    const loc = mlir.Location.unknown(ctx);
    const args = mlir.Block.init(&.{}, &.{});
    defer args.deinit();
    const float_type = mlir.Type.vector(&.{ 2, 8 }, .float(ctx, .f32));
    const int_type = mlir.Type.vector(&.{ 2, 8 }, .int(ctx, .i32));
    const floats = args.addArgument(float_type, loc);
    const integers = args.addArgument(int_type, loc);
    const mask = args.addArgument(mlir.Type.vector(&.{8}, .int(ctx, .i1)), loc);
    const float_min = all_reduce(ctx, floats, 1, .min, float_type, loc);
    defer float_min.deinit();
    const int_max = all_reduce(ctx, integers, 1, .max, int_type, loc);
    defer int_max.deinit();
    try std.testing.expect(float_min.attributeByName("kind").?.eql(try mlir.Attribute.parse(ctx, "#tpu.reduction_kind<minf>")));
    try std.testing.expect(int_max.attributeByName("kind").?.eql(try mlir.Attribute.parse(ctx, "#tpu.reduction_kind<maxsi>")));

    const int_scan = scan(ctx, integers, .max, mask, int_type, loc);
    defer int_scan.deinit();
    try std.testing.expect(int_scan.verify());
    try std.testing.expect(int_scan.attributeByName("kind").?.eql(try mlir.Attribute.parse(ctx, "#tpu.reduction_kind<maxui>")));
    try std.testing.expectEqual(@as(i64, 1), int_scan.attributeByName("dimension").?.isA(mlir.IntegerAttribute).?.value(i64));
    const float_scan = scan(ctx, floats, .min, null, float_type, loc);
    defer float_scan.deinit();
    try std.testing.expect(float_scan.attributeByName("kind").?.eql(try mlir.Attribute.parse(ctx, "#tpu.reduction_kind<minf>")));
}

fn expectPrints(expected: []const u8, value: anytype) !void {
    var buf: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try w.print("{f}", .{value});
    try std.testing.expectEqualStrings(expected, w.buffered());
}

fn expectEnumAttr(comptime Attr: type, ctx: *mlir.Context) !void {
    const Enum = @FieldType(Attr.InitArgs, "value");
    inline for (std.meta.fields(Enum)) |field| {
        const value: Enum = @enumFromInt(field.value);
        const attr = try Attr.get(ctx, .{ .value = value });
        try std.testing.expectEqual(value, attr.getValue());
        try std.testing.expect(attr.attribute().isA(Attr) != null);
        try std.testing.expect(value.attribute(ctx).eql(attr.attribute()));
        try expectPrints("#tpu." ++ Attr.mnemonic ++ "<" ++ field.name ++ ">", attr);
    }
}

test "tpu types are built through the C API" {
    const ctx = try mlir.Context.init(.{});
    defer ctx.deinit();

    const semaphore = try SemaphoreType.get(ctx, .{});
    try std.testing.expect(semaphoreType(ctx).isA(SemaphoreType) != null);
    try std.testing.expect(semaphoreType(ctx).isA(DMASemaphoreType) == null);
    try expectPrints("!tpu.semaphore", semaphore);

    const dma_semaphore = try DMASemaphoreType.get(ctx, .{});
    try std.testing.expect(dmaSemaphoreType(ctx).isA(DMASemaphoreType) != null);
    try expectPrints("!tpu.dma_semaphore", dma_semaphore);
}

test "tpu enum attributes are built through the C API" {
    const ctx = try mlir.Context.init(.{});
    defer ctx.deinit();

    try expectEnumAttr(ReductionKindAttr, ctx);
    try expectEnumAttr(ContractPrecisionAttr, ctx);
    try expectEnumAttr(RoundingModeAttr, ctx);
    try expectEnumAttr(CoreTypeAttr, ctx);
    try expectEnumAttr(DimensionSemanticsAttr, ctx);
    try expectEnumAttr(PipelineModeAttr, ctx);
    try expectEnumAttr(RevisitModeAttr, ctx);
    try expectEnumAttr(MemorySpaceAttr, ctx);

    const smem_tc = try MemorySpaceAttr.get(ctx, .{ .value = .smem, .coreType = .sc_scalar_subcore });
    try std.testing.expectEqual(MemorySpace.smem, smem_tc.getValue());
    try std.testing.expectEqual(CoreType.sc_scalar_subcore, smem_tc.getCoreType().?);
    try expectPrints("#tpu.memory_space<smem, sc_scalar_subcore>", smem_tc);
    try std.testing.expectEqual(null, (try MemorySpaceAttr.get(ctx, .{ .value = .hbm })).getCoreType());
    // The dialect verifier's pairing rules: no core type on hbm, no scalar subcore on vmem.
    try std.testing.expectError(error.InvalidMlir, MemorySpaceAttr.get(ctx, .{ .value = .hbm, .coreType = .tc }));
    try std.testing.expectError(error.InvalidMlir, MemorySpaceAttr.get(ctx, .{ .value = .vmem, .coreType = .sc_scalar_subcore }));
}

test "tpu array attributes are built through the C API" {
    const ctx = try mlir.Context.init(.{});
    defer ctx.deinit();

    const window = try ElementWindowAttr.get(ctx, .{ .padLow = &.{ 0, 2 }, .padHigh = &.{ 1, 3 } });
    try std.testing.expectEqual(2, window.getNumPadLow());
    try std.testing.expectEqual(2, window.getPadLow(1));
    try std.testing.expectEqual(2, window.getNumPadHigh());
    try std.testing.expectEqual(3, window.getPadHigh(1));
    try expectPrints("#tpu.element_window<[0, 2], [1, 3]>", window);
    try std.testing.expect(elementWindowAttribute(ctx, &.{ 0, 2 }, &.{ 1, 3 }).eql(window.attribute()));

    const dims = dotDimensionNumbers(ctx, &.{1}, &.{0}, &.{0}, &.{1}, &.{ 0, 0, 1, 1 }, &.{}, &.{});
    try expectPrints("#tpu.dot_dimension_numbers<[1], [0], [0], [1], [0, 0, 1, 1], [], []>", dims);

    const batched = try DotDimensionNumbersAttr.get(ctx, .{
        .lhsContractingDims = &.{2},
        .rhsContractingDims = &.{1},
        .lhsNonContractingDims = &.{1},
        .outputDimOrder = &.{ 0, 0, 0, 1 },
        .lhsBatchDims = &.{0},
        .rhsBatchDims = &.{0},
    });
    try std.testing.expectEqual(0, batched.getNumRhsNonContractingDims());
    try std.testing.expectEqual(4, batched.getNumOutputDimOrder());
    try std.testing.expectEqual(1, batched.getOutputDimOrder(3));
    try std.testing.expectEqual(0, batched.getRhsBatchDims(0));
    try expectPrints("#tpu.dot_dimension_numbers<[2], [1], [1], [], [0, 0, 0, 1], [0], [0]>", batched);
}
