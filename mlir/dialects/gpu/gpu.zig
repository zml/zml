const c = @import("c");
const mlir = @import("mlir");
const stdx = @import("stdx");

pub fn registerDialect(registry: *mlir.DialectRegistry) void {
    registry.registerDialect("gpu");
}

/// A dimension, either `x`, `y` or `z`.
pub const Dim = enum(u32) {
    x = 0,
    y = 1,
    z = 2,
};

/// Indexing modes supported by `gpu.shuffle`.
pub const ShuffleMode = enum(u32) {
    xor = 0,
    down = 1,
    up = 2,
    idx = 3,
};

/// `#gpu<dim ...>`.
pub const DimensionAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsAGPUDimension;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "dim";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: Dim,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirGPUDimensionAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) Dim {
        return @fromBackingInt(@intCast(c.mlirGPUDimensionAttrGetValue(self.ptr())));
    }
};

/// `#gpu<shuffle_mode ...>`.
pub const ShuffleModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsAGPUShuffleMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "shuffle_mode";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: ShuffleMode,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirGPUShuffleModeAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) ShuffleMode {
        return @fromBackingInt(@intCast(c.mlirGPUShuffleModeAttrGetValue(self.ptr())));
    }
};

pub fn module(ctx: *mlir.Context, name: []const u8, body: *mlir.Block, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "gpu.module", .{
        .blocks = &.{body},
        .attributes = &.{
            .named(ctx, "sym_name", .string(ctx, name)),
        },
        .location = location,
        .verify = false,
    });
}

pub const FuncArgs = struct {
    name: []const u8,
    args: []const *const mlir.Type,
    block: *mlir.Block,
    location: *const mlir.Location,
    kernel: bool = true,
    extra_attributes: []const mlir.NamedAttribute = &.{},
};

/// `gpu.func @name(args) kernel { ... gpu.return }`.
pub fn func(ctx: *mlir.Context, args: FuncArgs) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 16) = .empty;
    attrs.appendSliceAssumeCapacity(&.{
        .named(ctx, "sym_name", .string(ctx, args.name)),
        .named(ctx, "function_type", .typeAttr(.function(ctx, args.args, &.{}))),
    });
    if (args.kernel) attrs.appendAssumeCapacity(.named(ctx, "gpu.kernel", .unit(ctx)));
    attrs.appendSliceAssumeCapacity(args.extra_attributes);
    return mlir.Operation.make(ctx, "gpu.func", .{
        .blocks = &.{args.block},
        .attributes = attrs.constSlice(),
        .location = args.location,
        .verify = false,
    });
}

pub fn return_(ctx: *mlir.Context, values: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "gpu.return", .{
        .operands = .{ .flat = values },
        .location = location,
        .verify = false,
    });
}

fn idOp(ctx: *mlir.Context, comptime name: []const u8, dim: Dim, location: *const mlir.Location) *mlir.Operation {
    const dimension = DimensionAttr.get(ctx, .{ .value = dim }) catch unreachable;
    return mlir.Operation.make(ctx, name, .{
        .results = .{ .flat = &.{.index(ctx)} },
        .attributes = &.{.named(ctx, "dimension", dimension.attribute())},
        .location = location,
    });
}

pub fn thread_id(ctx: *mlir.Context, dim: Dim, location: *const mlir.Location) *mlir.Operation {
    return idOp(ctx, "gpu.thread_id", dim, location);
}

pub fn block_id(ctx: *mlir.Context, dim: Dim, location: *const mlir.Location) *mlir.Operation {
    return idOp(ctx, "gpu.block_id", dim, location);
}

pub fn block_dim(ctx: *mlir.Context, dim: Dim, location: *const mlir.Location) *mlir.Operation {
    return idOp(ctx, "gpu.block_dim", dim, location);
}

pub fn grid_dim(ctx: *mlir.Context, dim: Dim, location: *const mlir.Location) *mlir.Operation {
    return idOp(ctx, "gpu.grid_dim", dim, location);
}

/// `gpu.shuffle`: `value` from lane `offset` (per `mode`) within `width` lanes,
/// and whether that lane was valid (i1).
pub fn shuffle(ctx: *mlir.Context, value: *const mlir.Value, offset: *const mlir.Value, width: *const mlir.Value, mode: ShuffleMode, location: *const mlir.Location) *mlir.Operation {
    const mode_attr = ShuffleModeAttr.get(ctx, .{ .value = mode }) catch unreachable;
    return mlir.Operation.make(ctx, "gpu.shuffle", .{
        .operands = .{ .flat = &.{ value, offset, width } },
        .results = .{ .flat = &.{ value.type_(), .int(ctx, .i1) } },
        .attributes = &.{.named(ctx, "mode", mode_attr.attribute())},
        .location = location,
    });
}

pub fn barrier(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "gpu.barrier", .{ .location = location });
}
