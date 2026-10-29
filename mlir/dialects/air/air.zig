//! Metal AIR intrinsics and kernel ABI annotations. A builder emits the
//! `air.*` intrinsic it names, minus the type suffix its types select:
//! `fast_exp` on f32 is `@air.fast_exp.f32`, `max_u` on `vector<2xi32>` is
//! `@air.max.u.v2i32`. Arithmetic, memory and control flow use the arith,
//! llvm and scf dialects, with pointers as `!llvm.ptr` in AIR's address spaces.
const std = @import("std");

const arith = @import("mlir/dialects/arith");
const c = @import("c");
const func = @import("mlir/dialects/func");
const mlir = @import("mlir");

pub fn registerDialect(registry: *mlir.DialectRegistry) void {
    registry.registerDialect("air");
}

pub const Builtin = enum(u32) {
    thread_position_in_grid,
    threadgroup_position_in_grid,
    thread_position_in_threadgroup,
    threads_per_threadgroup,
    threadgroups_per_grid,
    threads_per_grid,
    thread_index_in_simdgroup,
    simdgroup_index_in_threadgroup,
    threads_per_simdgroup,
    simdgroups_per_threadgroup,
    thread_index_in_threadgroup,
};

pub const AddressSpace = enum(u32) { thread = 0, device = 1, constant = 2, threadgroup = 3 };

pub const Access = enum(u32) { read, read_write };

pub const MemFlags = packed struct(u32) {
    device: bool = false,
    threadgroup: bool = false,
    texture: bool = false,
    _padding: u29 = 0,

    pub const none: MemFlags = .{};
};

/// `#air.builtin`: Kernel argument builtin
pub const BuiltinAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsAMetalAIRBuiltin;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub const InitArgs = struct { value: Builtin };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirMetalAIRBuiltinAttrGet(ctx.ptr(), @backingInt(args.value)).ptr orelse return error.InvalidMlir);
    }

    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }

    pub fn value(self: *const Self) Builtin {
        return @fromBackingInt(@intCast(c.mlirMetalAIRBuiltinAttrGetValue(self.ptr())));
    }
};

/// `#air.mem_flags`: Memory a barrier orders
pub const MemFlagsAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsAMetalAIRMemFlags;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub const InitArgs = struct { value: MemFlags };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirMetalAIRMemFlagsAttrGet(ctx.ptr(), @bitCast(args.value)).ptr orelse return error.InvalidMlir);
    }

    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }

    pub fn value(self: *const Self) MemFlags {
        return @bitCast(c.mlirMetalAIRMemFlagsAttrGetValue(self.ptr()));
    }
};

/// `#air.struct_field`: Field of a constant buffer struct
pub const StructFieldAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsAMetalAIRStructField;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub const InitArgs = struct {
        offset: u64,
        size: u64,
        /// The array length, zero for a non-array field.
        count: u64 = 0,
        typeName: []const u8,
        name: []const u8,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirMetalAIRStructFieldAttrGet(ctx.ptr(), args.offset, args.size, args.count, mlir.stringRef(args.typeName), mlir.stringRef(args.name)).ptr orelse return error.InvalidMlir);
    }

    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }

    pub fn offset(self: *const Self) u64 {
        return c.mlirMetalAIRStructFieldAttrGetOffset(self.ptr());
    }

    pub fn size(self: *const Self) u64 {
        return c.mlirMetalAIRStructFieldAttrGetSize(self.ptr());
    }

    pub fn count(self: *const Self) u64 {
        return c.mlirMetalAIRStructFieldAttrGetCount(self.ptr());
    }

    pub fn typeName(self: *const Self) []const u8 {
        return mlir.string(c.mlirMetalAIRStructFieldAttrGetTypeName(self.ptr()));
    }

    pub fn name(self: *const Self) []const u8 {
        return mlir.string(c.mlirMetalAIRStructFieldAttrGetName(self.ptr()));
    }
};

/// `#air.buffer`: Kernel buffer argument
pub const BufferAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsAMetalAIRBuffer;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub const InitArgs = struct {
        locationIndex: u32,
        addressSpace: AddressSpace,
        access: Access,
        elementType: *const mlir.Type,
        typeName: []const u8,
        typeSize: u64,
        typeAlignment: u64,
        /// The size of a constant reference; zero omits it.
        bufferSize: u64 = 0,
        noalias_: bool = false,
        fields: []const *const StructFieldAttr = &.{},
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirMetalAIRBufferAttrGet(
            ctx.ptr(),
            args.locationIndex,
            @backingInt(args.addressSpace),
            @backingInt(args.access),
            args.elementType.ptr(),
            mlir.stringRef(args.typeName),
            args.typeSize,
            args.typeAlignment,
            args.bufferSize,
            args.noalias_,
            @intCast(args.fields.len),
            @ptrCast(args.fields.ptr),
        ).ptr orelse return error.InvalidMlir);
    }

    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }

    pub fn locationIndex(self: *const Self) u32 {
        return c.mlirMetalAIRBufferAttrGetLocationIndex(self.ptr());
    }

    pub fn addressSpace(self: *const Self) AddressSpace {
        return @fromBackingInt(@intCast(c.mlirMetalAIRBufferAttrGetAddressSpace(self.ptr())));
    }

    pub fn access(self: *const Self) Access {
        return @fromBackingInt(@intCast(c.mlirMetalAIRBufferAttrGetAccess(self.ptr())));
    }

    pub fn elementType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirMetalAIRBufferAttrGetElementType(self.ptr()).ptr);
    }

    pub fn typeName(self: *const Self) []const u8 {
        return mlir.string(c.mlirMetalAIRBufferAttrGetTypeName(self.ptr()));
    }

    pub fn typeSize(self: *const Self) u64 {
        return c.mlirMetalAIRBufferAttrGetTypeSize(self.ptr());
    }

    pub fn typeAlignment(self: *const Self) u64 {
        return c.mlirMetalAIRBufferAttrGetTypeAlignment(self.ptr());
    }

    pub fn bufferSize(self: *const Self) u64 {
        return c.mlirMetalAIRBufferAttrGetBufferSize(self.ptr());
    }

    pub fn noalias_(self: *const Self) bool {
        return c.mlirMetalAIRBufferAttrGetNoalias(self.ptr());
    }

    pub fn numFields(self: *const Self) usize {
        return @intCast(c.mlirMetalAIRBufferAttrGetNumFields(self.ptr()));
    }

    pub fn field(self: *const Self, i: usize) *const StructFieldAttr {
        return @ptrCast(c.mlirMetalAIRBufferAttrGetField(self.ptr(), @intCast(i)).ptr);
    }
};

pub const ArgumentOpts = struct {
    /// `air.arg_name`, the argument's name in reflection.
    name: ?[]const u8 = null,
};

/// The argument dictionary of a kernel argument bound to `builtin`.
pub fn builtinArgumentAttr(ctx: *mlir.Context, builtin: Builtin, opts: ArgumentOpts) mlir.Error!*const mlir.Attribute {
    return argumentAttr(ctx, .named(ctx, "air.builtin", (try BuiltinAttr.get(ctx, .{ .value = builtin })).attribute()), opts);
}

/// The argument dictionary of a kernel buffer argument.
pub fn bufferArgumentAttr(ctx: *mlir.Context, args: BufferAttr.InitArgs, opts: ArgumentOpts) mlir.Error!*const mlir.Attribute {
    return argumentAttr(ctx, .named(ctx, "air.buffer", (try BufferAttr.get(ctx, args)).attribute()), opts);
}

fn argumentAttr(ctx: *mlir.Context, annotation: mlir.NamedAttribute, opts: ArgumentOpts) *const mlir.Attribute {
    if (opts.name) |name| return .dict(ctx, &.{ annotation, .named(ctx, "air.arg_name", .string(ctx, name)) });
    return .dict(ctx, &.{annotation});
}

/// Marks `module` as compiled with fast math (`air.compile.fast_math_enable`).
pub fn enableFastMath(ctx: *mlir.Context, module: *mlir.Operation) void {
    module.setAttributeByName("air.fast_math_enable", .unit(ctx));
}

pub const KernelArgs = struct {
    name: []const u8,
    block: *mlir.Block,
    args_attributes: []const *const mlir.Attribute,
    location: *const mlir.Location,
    verify: bool = true,
};

/// A `func.func` with `air.kernel`, one argument dictionary per block
/// argument. Takes ownership of `block`; set `verify = false` while the body
/// is incomplete.
pub fn kernel(ctx: *mlir.Context, args: KernelArgs) *mlir.Operation {
    std.debug.assert(args.args_attributes.len == args.block.numArguments());
    return func.func(ctx, .{
        .name = args.name,
        .block = args.block,
        .args_attributes = args.args_attributes,
        .results = &.{},
        .extra_attributes = &.{.named(ctx, "air.kernel", .unit(ctx))},
        .location = args.location,
        .verify = args.verify,
    });
}

pub const FloatOpts = struct {
    fastmath: arith.FastMathFlags = .none,
};

/// The `fastmath` attribute of a float op, stored in the caller's `buf`.
fn fastmathAttrs(buf: *[1]mlir.NamedAttribute, ctx: *mlir.Context, fastmath: arith.FastMathFlags) []const mlir.NamedAttribute {
    if (fastmath == arith.FastMathFlags.none) return &.{};
    const attr = arith.FastMathAttr.get(ctx, .{ .value = fastmath }) catch unreachable;
    buf[0] = .named(ctx, "fastmath", attr.attribute());
    return buf;
}

fn floatUnary(comptime name: []const u8) type {
    return struct {
        pub fn call(ctx: *mlir.Context, x: *const mlir.Value, opts: FloatOpts, location: *const mlir.Location) *mlir.Operation {
            var attrs: [1]mlir.NamedAttribute = undefined;
            return mlir.Operation.make(ctx, name, .{
                .operands = .{ .flat = &.{x} },
                .results = .{ .flat = &.{x.type_()} },
                .attributes = fastmathAttrs(&attrs, ctx, opts.fastmath),
                .location = location,
            });
        }
    };
}

fn floatBinary(comptime name: []const u8) type {
    return struct {
        pub fn call(ctx: *mlir.Context, x: *const mlir.Value, y: *const mlir.Value, opts: FloatOpts, location: *const mlir.Location) *mlir.Operation {
            var attrs: [1]mlir.NamedAttribute = undefined;
            return mlir.Operation.make(ctx, name, .{
                .operands = .{ .flat = &.{ x, y } },
                .results = .{ .flat = &.{x.type_()} },
                .attributes = fastmathAttrs(&attrs, ctx, opts.fastmath),
                .location = location,
            });
        }
    };
}

fn floatTernary(comptime name: []const u8) type {
    return struct {
        pub fn call(ctx: *mlir.Context, x: *const mlir.Value, y: *const mlir.Value, z: *const mlir.Value, opts: FloatOpts, location: *const mlir.Location) *mlir.Operation {
            var attrs: [1]mlir.NamedAttribute = undefined;
            return mlir.Operation.make(ctx, name, .{
                .operands = .{ .flat = &.{ x, y, z } },
                .results = .{ .flat = &.{x.type_()} },
                .attributes = fastmathAttrs(&attrs, ctx, opts.fastmath),
                .location = location,
            });
        }
    };
}

fn intUnary(comptime name: []const u8) type {
    return struct {
        pub fn call(ctx: *mlir.Context, x: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
            return mlir.Operation.make(ctx, name, .{
                .operands = .{ .flat = &.{x} },
                .results = .{ .flat = &.{x.type_()} },
                .location = location,
            });
        }
    };
}

fn intBinary(comptime name: []const u8) type {
    return struct {
        pub fn call(ctx: *mlir.Context, x: *const mlir.Value, y: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
            return mlir.Operation.make(ctx, name, .{
                .operands = .{ .flat = &.{ x, y } },
                .results = .{ .flat = &.{x.type_()} },
                .location = location,
            });
        }
    };
}

fn intTernary(comptime name: []const u8) type {
    return struct {
        pub fn call(ctx: *mlir.Context, x: *const mlir.Value, y: *const mlir.Value, z: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
            return mlir.Operation.make(ctx, name, .{
                .operands = .{ .flat = &.{ x, y, z } },
                .results = .{ .flat = &.{x.type_()} },
                .location = location,
            });
        }
    };
}

fn floatShuffle(comptime name: []const u8) type {
    return struct {
        pub fn call(ctx: *mlir.Context, data: *const mlir.Value, lane: *const mlir.Value, opts: FloatOpts, location: *const mlir.Location) *mlir.Operation {
            var attrs: [1]mlir.NamedAttribute = undefined;
            return mlir.Operation.make(ctx, name, .{
                .operands = .{ .flat = &.{ data, lane } },
                .results = .{ .flat = &.{data.type_()} },
                .attributes = fastmathAttrs(&attrs, ctx, opts.fastmath),
                .location = location,
            });
        }
    };
}

fn intShuffle(comptime name: []const u8) type {
    return struct {
        pub fn call(ctx: *mlir.Context, data: *const mlir.Value, lane: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
            return mlir.Operation.make(ctx, name, .{
                .operands = .{ .flat = &.{ data, lane } },
                .results = .{ .flat = &.{data.type_()} },
                .location = location,
            });
        }
    };
}

fn convertOp(comptime name: []const u8) type {
    return struct {
        pub fn call(ctx: *mlir.Context, x: *const mlir.Value, result_type: *const mlir.Type, opts: FloatOpts, location: *const mlir.Location) *mlir.Operation {
            var attrs: [1]mlir.NamedAttribute = undefined;
            return mlir.Operation.make(ctx, name, .{
                .operands = .{ .flat = &.{x} },
                .results = .{ .flat = &.{result_type} },
                .attributes = fastmathAttrs(&attrs, ctx, opts.fastmath),
                .location = location,
            });
        }
    };
}

fn barrier(comptime name: []const u8) type {
    return struct {
        pub fn call(ctx: *mlir.Context, mem_flags: MemFlags, location: *const mlir.Location) *mlir.Operation {
            const flags = MemFlagsAttr.get(ctx, .{ .value = mem_flags }) catch unreachable;
            return mlir.Operation.make(ctx, name, .{
                .attributes = &.{.named(ctx, "mem_flags", flags.attribute())},
                .location = location,
            });
        }
    };
}

pub const exp = floatUnary("air.exp").call;
pub const fast_exp = floatUnary("air.fast_exp").call;
pub const exp2 = floatUnary("air.exp2").call;
pub const fast_exp2 = floatUnary("air.fast_exp2").call;
pub const log = floatUnary("air.log").call;
pub const fast_log = floatUnary("air.fast_log").call;
pub const log2 = floatUnary("air.log2").call;
pub const fast_log2 = floatUnary("air.fast_log2").call;
pub const tanh = floatUnary("air.tanh").call;
pub const fast_tanh = floatUnary("air.fast_tanh").call;
pub const sqrt = floatUnary("air.sqrt").call;
pub const fast_sqrt = floatUnary("air.fast_sqrt").call;
pub const rsqrt = floatUnary("air.rsqrt").call;
pub const fast_rsqrt = floatUnary("air.fast_rsqrt").call;
pub const fabs = floatUnary("air.fabs").call;
pub const fast_fabs = floatUnary("air.fast_fabs").call;
pub const ceil = floatUnary("air.ceil").call;
pub const fast_ceil = floatUnary("air.fast_ceil").call;
pub const floor = floatUnary("air.floor").call;
pub const fast_floor = floatUnary("air.fast_floor").call;
pub const round = floatUnary("air.round").call;
pub const fast_round = floatUnary("air.fast_round").call;
pub const trunc = floatUnary("air.trunc").call;
pub const fast_trunc = floatUnary("air.fast_trunc").call;
pub const sin = floatUnary("air.sin").call;
pub const fast_sin = floatUnary("air.fast_sin").call;
pub const cos = floatUnary("air.cos").call;
pub const fast_cos = floatUnary("air.fast_cos").call;
pub const fmax = floatBinary("air.fmax").call;
pub const fast_fmax = floatBinary("air.fast_fmax").call;
pub const fmin = floatBinary("air.fmin").call;
pub const fast_fmin = floatBinary("air.fast_fmin").call;
pub const pow = floatBinary("air.pow").call;
pub const fast_pow = floatBinary("air.fast_pow").call;
pub const fmod = floatBinary("air.fmod").call;
pub const fast_fmod = floatBinary("air.fast_fmod").call;
pub const fma = floatTernary("air.fma").call;
pub const clamp = floatTernary("air.clamp").call;
pub const fast_clamp = floatTernary("air.fast_clamp").call;
pub const abs_s = intUnary("air.abs.s").call;
pub const max_s = intBinary("air.max.s").call;
pub const max_u = intBinary("air.max.u").call;
pub const min_s = intBinary("air.min.s").call;
pub const min_u = intBinary("air.min.u").call;
pub const clamp_s = intTernary("air.clamp.s").call;
pub const clamp_u = intTernary("air.clamp.u").call;
pub const simd_sum = floatUnary("air.simd_sum").call;
pub const simd_sum_s = intUnary("air.simd_sum.s").call;
pub const simd_sum_u = intUnary("air.simd_sum.u").call;
pub const simd_max = floatUnary("air.simd_max").call;
pub const simd_max_s = intUnary("air.simd_max.s").call;
pub const simd_max_u = intUnary("air.simd_max.u").call;
pub const simd_min = floatUnary("air.simd_min").call;
pub const simd_min_s = intUnary("air.simd_min.s").call;
pub const simd_min_u = intUnary("air.simd_min.u").call;
pub const simd_prefix_exclusive_sum = floatUnary("air.simd_prefix_exclusive_sum").call;
pub const simd_shuffle = floatShuffle("air.simd_shuffle").call;
pub const simd_shuffle_s = intShuffle("air.simd_shuffle.s").call;
pub const simd_shuffle_u = intShuffle("air.simd_shuffle.u").call;
pub const simd_shuffle_xor = floatShuffle("air.simd_shuffle_xor").call;
pub const simd_shuffle_xor_s = intShuffle("air.simd_shuffle_xor.s").call;
pub const simd_shuffle_xor_u = intShuffle("air.simd_shuffle_xor.u").call;
pub const simd_shuffle_down = floatShuffle("air.simd_shuffle_down").call;
pub const simd_shuffle_down_s = intShuffle("air.simd_shuffle_down.s").call;
pub const simd_shuffle_down_u = intShuffle("air.simd_shuffle_down.u").call;
pub const simd_broadcast = floatShuffle("air.simd_broadcast").call;
pub const simd_broadcast_s = intShuffle("air.simd_broadcast.s").call;
pub const simd_broadcast_u = intShuffle("air.simd_broadcast.u").call;
pub const simd_any = intUnary("air.simd_any").call;
pub const simd_all = intUnary("air.simd_all").call;
pub const convert = convertOp("air.convert").call;
pub const convert_s = convertOp("air.convert.s").call;
pub const convert_u = convertOp("air.convert.u").call;
pub const wg_barrier = barrier("air.wg.barrier").call;
pub const simdgroup_barrier = barrier("air.simdgroup.barrier").call;

pub fn dot(ctx: *mlir.Context, x: *const mlir.Value, y: *const mlir.Value, opts: FloatOpts, location: *const mlir.Location) *mlir.Operation {
    var attrs: [1]mlir.NamedAttribute = undefined;
    return mlir.Operation.make(ctx, "air.dot", .{
        .operands = .{ .flat = &.{ x, y } },
        .results = .{ .flat = &.{x.type_().isA(mlir.ShapedType).?.elementType()} },
        .attributes = fastmathAttrs(&attrs, ctx, opts.fastmath),
        .location = location,
    });
}

/// `simdgroup_load`: `extents`, `strides` and `origin` are `vector<2xi64>`.
pub fn simdgroup_matrix_8x8_load(ctx: *mlir.Context, pointer: *const mlir.Value, extents: *const mlir.Value, strides: *const mlir.Value, origin: *const mlir.Value, result_type: *const mlir.Type, opts: FloatOpts, location: *const mlir.Location) *mlir.Operation {
    var attrs: [1]mlir.NamedAttribute = undefined;
    return mlir.Operation.make(ctx, "air.simdgroup_matrix_8x8_load", .{
        .operands = .{ .flat = &.{ pointer, extents, strides, origin } },
        .results = .{ .flat = &.{result_type} },
        .attributes = fastmathAttrs(&attrs, ctx, opts.fastmath),
        .location = location,
    });
}

pub fn simdgroup_matrix_8x8_store(ctx: *mlir.Context, value: *const mlir.Value, pointer: *const mlir.Value, extents: *const mlir.Value, strides: *const mlir.Value, origin: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "air.simdgroup_matrix_8x8_store", .{
        .operands = .{ .flat = &.{ value, pointer, extents, strides, origin } },
        .location = location,
    });
}

/// `simdgroup_multiply_accumulate`: `a * b + c`.
pub fn simdgroup_matrix_8x8_multiply_accumulate(ctx: *mlir.Context, a: *const mlir.Value, b: *const mlir.Value, c_: *const mlir.Value, result_type: *const mlir.Type, opts: FloatOpts, location: *const mlir.Location) *mlir.Operation {
    var attrs: [1]mlir.NamedAttribute = undefined;
    return mlir.Operation.make(ctx, "air.simdgroup_matrix_8x8_multiply_accumulate", .{
        .operands = .{ .flat = &.{ a, b, c_ } },
        .results = .{ .flat = &.{result_type} },
        .attributes = fastmathAttrs(&attrs, ctx, opts.fastmath),
        .location = location,
    });
}
