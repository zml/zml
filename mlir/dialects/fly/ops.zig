//! `fly` operation builders, generated from FlyDSL's `include/flydsl/Dialect/Fly/IR/FlyOps.td` at
//! 1941889400621f1fc3d8f03bada1a4e7380cdf7e; do not edit, see REGENERATING.md.
//!
//! Builders take the operands, then the result types (only for operations
//! that do not infer them), then the attributes, then the location.
//! Parameters are the ODS names in snake case, with a trailing underscore
//! where needed. Pass `null` for absent optional operands or attributes and
//! `&.{}` for empty variadic ones; variable-length operand groups of
//! `AttrSizedOperandSegments` operations get their segment sizes from
//! `Operation.make`. Attributes use `mlir.Attribute`.

const std = @import("std");

const mlir = @import("mlir");
const stdx = @import("stdx");

const make = @import("fly.zig").make;

/// Every operation bound here, for the registration test.
pub const names = [_][]const u8{
    "fly.add_offset",
    "fly.append",
    "fly.apply_swizzle",
    "fly.assume",
    "fly.atom.set_value",
    "fly.blocked_product",
    "fly.ceil_div",
    "fly.coalesce",
    "fly.complement",
    "fly.composed_get_inner",
    "fly.composed_get_offset",
    "fly.composed_get_outer",
    "fly.composition",
    "fly.coprofile",
    "fly.copy",
    "fly.copy_atom_call",
    "fly.copy_atom_call_ssa",
    "fly.coshape",
    "fly.cosize",
    "fly.crd2idx",
    "fly.decomposition",
    "fly.dice",
    "fly.elem_less",
    "fly.equal",
    "fly.extract_aligned_pointer_as_index",
    "fly.flat_divide",
    "fly.flat_product",
    "fly.gemm",
    "fly.get",
    "fly.get_1d_coord",
    "fly.get_copy_atom",
    "fly.get_dyn_shared",
    "fly.get_flat_coord",
    "fly.get_iter",
    "fly.get_layout",
    "fly.get_leaves",
    "fly.get_mma_atom",
    "fly.get_scalar",
    "fly.get_shape",
    "fly.get_stride",
    "fly.group",
    "fly.idx2crd",
    "fly.int_tuple_add",
    "fly.int_tuple_div",
    "fly.int_tuple_mod",
    "fly.int_tuple_mul",
    "fly.int_tuple_product",
    "fly.int_tuple_product_each",
    "fly.int_tuple_product_like",
    "fly.int_tuple_sub",
    "fly.inttoptr",
    "fly.left_inverse",
    "fly.logical_divide",
    "fly.logical_product",
    "fly.make_composed_layout",
    "fly.make_coord",
    "fly.make_copy_atom",
    "fly.make_fragment_layout_like",
    "fly.make_fragment_like",
    "fly.make_identity_layout",
    "fly.make_int_tuple",
    "fly.make_layout",
    "fly.make_layout_like",
    "fly.make_mma_atom",
    "fly.make_ordered_layout",
    "fly.make_ptr",
    "fly.make_shape",
    "fly.make_stride",
    "fly.make_tiled_copy",
    "fly.make_tiled_mma",
    "fly.make_view",
    "fly.memref.alloca",
    "fly.memref.load",
    "fly.memref.load_vec",
    "fly.memref.store",
    "fly.memref.store_vec",
    "fly.mma.make_fragment",
    "fly.mma_atom_call",
    "fly.mma_atom_call_ssa",
    "fly.prepend",
    "fly.print",
    "fly.ptr.load",
    "fly.ptr.store",
    "fly.ptrtoint",
    "fly.raked_product",
    "fly.recast_iter",
    "fly.recast_layout",
    "fly.right_inverse",
    "fly.select",
    "fly.shape_div",
    "fly.size",
    "fly.slice",
    "fly.static",
    "fly.take",
    "fly.tile_to_shape",
    "fly.tiled_copy.partition_dst",
    "fly.tiled_copy.partition_src",
    "fly.tiled_copy.retile",
    "fly.tiled_divide",
    "fly.tiled_mma.partition",
    "fly.tiled_mma.partition_shape",
    "fly.tiled_product",
    "fly.to_llvm_ptr",
    "fly.zipped_divide",
    "fly.zipped_product",
};

/// `fly.add_offset`. Result types are inferred.
pub fn add_offset(ctx: *mlir.Context, ptr: *const mlir.Value, offset: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.add_offset", .{
        .operands = .{ .flat = &.{ ptr, offset } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.append`. Result types are inferred.
pub fn append(ctx: *mlir.Context, tuple: *const mlir.Value, elem: *const mlir.Value, n: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (n) |present| attributes.appendAssumeCapacity(.named(ctx, "n", present));
    return make(ctx, "fly.append", .{
        .operands = .{ .flat = &.{ tuple, elem } },
        .attributes = attributes.constSlice(),
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.apply_swizzle`. Result types are inferred.
pub fn apply_swizzle(ctx: *mlir.Context, ptr: *const mlir.Value, swizzle: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.apply_swizzle", .{
        .operands = .{ .flat = &.{ ptr, swizzle } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.assume`. Result types are explicit.
pub fn assume(ctx: *mlir.Context, dst: *const mlir.Value, src: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.assume", .{
        .operands = .{ .flat = &.{ dst, src } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `fly.atom.set_value`. Result types are inferred.
pub fn atom_set_value(ctx: *mlir.Context, atom: *const mlir.Value, value: *const mlir.Value, field: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.atom.set_value", .{
        .operands = .{ .flat = &.{ atom, value } },
        .attributes = &.{
            .named(ctx, "field", field),
        },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.blocked_product`. Result types are inferred.
pub fn blocked_product(ctx: *mlir.Context, layout: *const mlir.Value, tile: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.blocked_product", .{
        .operands = .{ .flat = &.{ layout, tile } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.ceil_div`. Result types are inferred.
pub fn ceil_div(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.ceil_div", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.coalesce`. Result types are inferred.
pub fn coalesce(ctx: *mlir.Context, layout: *const mlir.Value, pattern: ?*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.coalesce", .{
        .operands = .{ .flat = if (pattern) |present| &.{ layout, present } else &.{layout} },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.complement`. Result types are inferred.
pub fn complement(ctx: *mlir.Context, layout: *const mlir.Value, codomain_size: ?*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.complement", .{
        .operands = .{ .flat = if (codomain_size) |present| &.{ layout, present } else &.{layout} },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.composed_get_inner`. Result types are inferred.
pub fn composed_get_inner(ctx: *mlir.Context, input: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.composed_get_inner", .{
        .operands = .{ .flat = &.{input} },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.composed_get_offset`. Result types are inferred.
pub fn composed_get_offset(ctx: *mlir.Context, input: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.composed_get_offset", .{
        .operands = .{ .flat = &.{input} },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.composed_get_outer`. Result types are inferred.
pub fn composed_get_outer(ctx: *mlir.Context, input: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.composed_get_outer", .{
        .operands = .{ .flat = &.{input} },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.composition`. Result types are inferred.
pub fn composition(ctx: *mlir.Context, outer: *const mlir.Value, inner: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.composition", .{
        .operands = .{ .flat = &.{ outer, inner } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.coprofile`. Result types are inferred.
pub fn coprofile(ctx: *mlir.Context, layout: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.coprofile", .{
        .operands = .{ .flat = &.{layout} },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.copy`. No results.
pub fn copy(ctx: *mlir.Context, copy_atom: *const mlir.Value, src: *const mlir.Value, dst: *const mlir.Value, pred: ?*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.copy", .{
        .operands = .{ .flat = if (pred) |present| &.{ copy_atom, src, dst, present } else &.{ copy_atom, src, dst } },
        .location = location,
    });
}

/// `fly.copy_atom_call`. No results.
pub fn copy_atom_call(ctx: *mlir.Context, copy_atom: *const mlir.Value, src: *const mlir.Value, dst: *const mlir.Value, pred: ?*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.copy_atom_call", .{
        .operands = .{ .flat = if (pred) |present| &.{ copy_atom, src, dst, present } else &.{ copy_atom, src, dst } },
        .location = location,
    });
}

/// `fly.copy_atom_call_ssa`. Result types are explicit.
pub fn copy_atom_call_ssa(ctx: *mlir.Context, copy_atom: *const mlir.Value, src: *const mlir.Value, dst: ?*const mlir.Value, pred: ?*const mlir.Value, results_types: []const *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.copy_atom_call_ssa", .{
        .operands = .{ .variadic = &.{
            &.{copy_atom},
            &.{src},
            if (dst) |present| &.{present} else &.{},
            if (pred) |present| &.{present} else &.{},
        } },
        .results = .{ .flat = results_types },
        .location = location,
    });
}

/// `fly.coshape`. Result types are inferred.
pub fn coshape(ctx: *mlir.Context, layout: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.coshape", .{
        .operands = .{ .flat = &.{layout} },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.cosize`. Result types are inferred.
pub fn cosize(ctx: *mlir.Context, layout: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.cosize", .{
        .operands = .{ .flat = &.{layout} },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.crd2idx`. Result types are inferred.
pub fn crd2idx(ctx: *mlir.Context, coord: *const mlir.Value, layout: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.crd2idx", .{
        .operands = .{ .flat = &.{ coord, layout } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.decomposition`. Result types are inferred.
pub fn decomposition(ctx: *mlir.Context, tensor: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.decomposition", .{
        .operands = .{ .flat = &.{tensor} },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.dice`. Result types are inferred.
pub fn dice(ctx: *mlir.Context, src: *const mlir.Value, coord: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.dice", .{
        .operands = .{ .flat = &.{ src, coord } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.elem_less`. Result types are explicit.
pub fn elem_less(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.elem_less", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `fly.equal`. Result types are explicit.
pub fn equal(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.equal", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `fly.extract_aligned_pointer_as_index`. Extract the raw pointer from a fly.memref. Result types are explicit.
pub fn extract_aligned_pointer_as_index(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.extract_aligned_pointer_as_index", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `fly.flat_divide`. Result types are inferred.
pub fn flat_divide(ctx: *mlir.Context, layout: *const mlir.Value, divisor: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.flat_divide", .{
        .operands = .{ .flat = &.{ layout, divisor } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.flat_product`. Result types are inferred.
pub fn flat_product(ctx: *mlir.Context, layout: *const mlir.Value, tile: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.flat_product", .{
        .operands = .{ .flat = &.{ layout, tile } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.gemm`. Fully expand tiled GEMM with variadic A and B operand groups. No results.
pub fn gemm(ctx: *mlir.Context, mma_atom: *const mlir.Value, d: *const mlir.Value, a: []const *const mlir.Value, b: []const *const mlir.Value, c: *const mlir.Value, traversal_layout: ?*const mlir.Value, traversal_order: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (traversal_order) |present| attributes.appendAssumeCapacity(.named(ctx, "traversalOrder", present));
    return make(ctx, "fly.gemm", .{
        .operands = .{ .variadic = &.{
            &.{mma_atom},
            &.{d},
            a,
            b,
            &.{c},
            if (traversal_layout) |present| &.{present} else &.{},
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `fly.get`. Result types are inferred.
pub fn get(ctx: *mlir.Context, input: *const mlir.Value, mode: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.get", .{
        .operands = .{ .flat = &.{input} },
        .attributes = &.{
            .named(ctx, "mode", mode),
        },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.get_1d_coord`. Result types are inferred.
pub fn get_1d_coord(ctx: *mlir.Context, index: *const mlir.Value, layout: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.get_1d_coord", .{
        .operands = .{ .flat = &.{ index, layout } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.get_copy_atom`. Extract the copy atom and its runtime state from a tiled copy. Result types are inferred.
pub fn get_copy_atom(ctx: *mlir.Context, tiled_copy: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.get_copy_atom", .{
        .operands = .{ .flat = &.{tiled_copy} },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.get_dyn_shared`. Result types are inferred.
pub fn get_dyn_shared(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.get_dyn_shared", .{
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.get_flat_coord`. Result types are inferred.
pub fn get_flat_coord(ctx: *mlir.Context, index: *const mlir.Value, layout: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.get_flat_coord", .{
        .operands = .{ .flat = &.{ index, layout } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.get_iter`. Result types are inferred.
pub fn get_iter(ctx: *mlir.Context, memref: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.get_iter", .{
        .operands = .{ .flat = &.{memref} },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.get_layout`. Result types are inferred.
pub fn get_layout(ctx: *mlir.Context, memref: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.get_layout", .{
        .operands = .{ .flat = &.{memref} },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.get_leaves`. Result types are inferred.
pub fn get_leaves(ctx: *mlir.Context, input: *const mlir.Value, dynamic_only: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (dynamic_only) |present| attributes.appendAssumeCapacity(.named(ctx, "dynamicOnly", present));
    return make(ctx, "fly.get_leaves", .{
        .operands = .{ .flat = &.{input} },
        .attributes = attributes.constSlice(),
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.get_mma_atom`. Extract the MMA atom and its runtime state from a tiled MMA. Result types are inferred.
pub fn get_mma_atom(ctx: *mlir.Context, tiled_mma: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.get_mma_atom", .{
        .operands = .{ .flat = &.{tiled_mma} },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.get_scalar`. Result types are inferred.
pub fn get_scalar(ctx: *mlir.Context, int_tuple: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.get_scalar", .{
        .operands = .{ .flat = &.{int_tuple} },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.get_shape`. Result types are inferred.
pub fn get_shape(ctx: *mlir.Context, layout: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.get_shape", .{
        .operands = .{ .flat = &.{layout} },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.get_stride`. Result types are inferred.
pub fn get_stride(ctx: *mlir.Context, layout: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.get_stride", .{
        .operands = .{ .flat = &.{layout} },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.group`. Result types are inferred.
pub fn group(ctx: *mlir.Context, tuple: *const mlir.Value, begin: *const mlir.Attribute, end: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.group", .{
        .operands = .{ .flat = &.{tuple} },
        .attributes = &.{
            .named(ctx, "begin", begin),
            .named(ctx, "end", end),
        },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.idx2crd`. Result types are inferred.
pub fn idx2crd(ctx: *mlir.Context, coord: *const mlir.Value, layout: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.idx2crd", .{
        .operands = .{ .flat = &.{ coord, layout } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.int_tuple_add`. Result types are inferred.
pub fn int_tuple_add(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.int_tuple_add", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.int_tuple_div`. Result types are inferred.
pub fn int_tuple_div(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.int_tuple_div", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.int_tuple_mod`. Result types are inferred.
pub fn int_tuple_mod(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.int_tuple_mod", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.int_tuple_mul`. Result types are inferred.
pub fn int_tuple_mul(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.int_tuple_mul", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.int_tuple_product`. Result types are inferred.
pub fn int_tuple_product(ctx: *mlir.Context, input: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.int_tuple_product", .{
        .operands = .{ .flat = &.{input} },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.int_tuple_product_each`. Result types are inferred.
pub fn int_tuple_product_each(ctx: *mlir.Context, input: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.int_tuple_product_each", .{
        .operands = .{ .flat = &.{input} },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.int_tuple_product_like`. Result types are inferred.
pub fn int_tuple_product_like(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.int_tuple_product_like", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.int_tuple_sub`. Result types are inferred.
pub fn int_tuple_sub(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.int_tuple_sub", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.inttoptr`. Result types are explicit.
pub fn inttoptr(ctx: *mlir.Context, src: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.inttoptr", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `fly.left_inverse`. Result types are inferred.
pub fn left_inverse(ctx: *mlir.Context, layout: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.left_inverse", .{
        .operands = .{ .flat = &.{layout} },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.logical_divide`. Result types are inferred.
pub fn logical_divide(ctx: *mlir.Context, layout: *const mlir.Value, divisor: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.logical_divide", .{
        .operands = .{ .flat = &.{ layout, divisor } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.logical_product`. Result types are inferred.
pub fn logical_product(ctx: *mlir.Context, layout: *const mlir.Value, tile: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.logical_product", .{
        .operands = .{ .flat = &.{ layout, tile } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.make_composed_layout`. Result types are inferred.
pub fn make_composed_layout(ctx: *mlir.Context, inner: *const mlir.Value, offset: *const mlir.Value, outer: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.make_composed_layout", .{
        .operands = .{ .flat = &.{ inner, offset, outer } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.make_coord`. Result types are explicit.
pub fn make_coord(ctx: *mlir.Context, dync_elems: []const *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.make_coord", .{
        .operands = .{ .flat = dync_elems },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `fly.make_copy_atom`. Result types are explicit.
pub fn make_copy_atom(ctx: *mlir.Context, args: []const *const mlir.Value, result_type: *const mlir.Type, val_bits: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.make_copy_atom", .{
        .operands = .{ .flat = args },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "valBits", val_bits),
        },
        .location = location,
    });
}

/// `fly.make_fragment_layout_like`. Result types are inferred.
pub fn make_fragment_layout_like(ctx: *mlir.Context, src: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.make_fragment_layout_like", .{
        .operands = .{ .flat = &.{src} },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.make_fragment_like`. Result types are inferred.
pub fn make_fragment_like(ctx: *mlir.Context, src: *const mlir.Value, dtype: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (dtype) |present| attributes.appendAssumeCapacity(.named(ctx, "dtype", present));
    return make(ctx, "fly.make_fragment_like", .{
        .operands = .{ .flat = &.{src} },
        .attributes = attributes.constSlice(),
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.make_identity_layout`. Result types are inferred.
pub fn make_identity_layout(ctx: *mlir.Context, shape: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.make_identity_layout", .{
        .operands = .{ .flat = &.{shape} },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.make_int_tuple`. Result types are explicit.
pub fn make_int_tuple(ctx: *mlir.Context, dync_elems: []const *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.make_int_tuple", .{
        .operands = .{ .flat = dync_elems },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `fly.make_layout`. Result types are inferred.
pub fn make_layout(ctx: *mlir.Context, shape: *const mlir.Value, stride: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.make_layout", .{
        .operands = .{ .flat = &.{ shape, stride } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.make_layout_like`. Result types are inferred.
pub fn make_layout_like(ctx: *mlir.Context, ref: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.make_layout_like", .{
        .operands = .{ .flat = &.{ref} },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.make_mma_atom`. Result types are explicit.
pub fn make_mma_atom(ctx: *mlir.Context, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.make_mma_atom", .{
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `fly.make_ordered_layout`. Result types are inferred.
pub fn make_ordered_layout(ctx: *mlir.Context, shape: *const mlir.Value, order: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.make_ordered_layout", .{
        .operands = .{ .flat = &.{ shape, order } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.make_ptr`. Result types are explicit.
pub fn make_ptr(ctx: *mlir.Context, args: []const *const mlir.Value, result_type: *const mlir.Type, dict_attrs: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (dict_attrs) |present| attributes.appendAssumeCapacity(.named(ctx, "dictAttrs", present));
    return make(ctx, "fly.make_ptr", .{
        .operands = .{ .flat = args },
        .results = .{ .flat = &.{result_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `fly.make_shape`. Result types are explicit.
pub fn make_shape(ctx: *mlir.Context, dync_elems: []const *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.make_shape", .{
        .operands = .{ .flat = dync_elems },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `fly.make_stride`. Result types are explicit.
pub fn make_stride(ctx: *mlir.Context, dync_elems: []const *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.make_stride", .{
        .operands = .{ .flat = dync_elems },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `fly.make_tiled_copy`. Result types are inferred.
pub fn make_tiled_copy(ctx: *mlir.Context, copy_atom: *const mlir.Value, layout_thr_val: *const mlir.Value, tile_mn: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.make_tiled_copy", .{
        .operands = .{ .flat = &.{ copy_atom, layout_thr_val, tile_mn } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.make_tiled_mma`. Result types are inferred.
pub fn make_tiled_mma(ctx: *mlir.Context, mma_atom: *const mlir.Value, atom_layout: *const mlir.Value, permutation: ?*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.make_tiled_mma", .{
        .operands = .{ .flat = if (permutation) |present| &.{ mma_atom, atom_layout, present } else &.{ mma_atom, atom_layout } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.make_view`. Result types are inferred.
pub fn make_view(ctx: *mlir.Context, iter: *const mlir.Value, layout: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.make_view", .{
        .operands = .{ .flat = &.{ iter, layout } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.memref.alloca`. Result types are explicit.
pub fn memref_alloca(ctx: *mlir.Context, layout: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.memref.alloca", .{
        .operands = .{ .flat = &.{layout} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `fly.memref.load`. Result types are inferred.
pub fn memref_load(ctx: *mlir.Context, memref: *const mlir.Value, indices: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.memref.load", .{
        .operands = .{ .flat = &.{ memref, indices } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.memref.load_vec`. Result types are inferred.
pub fn memref_load_vec(ctx: *mlir.Context, memref: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.memref.load_vec", .{
        .operands = .{ .flat = &.{memref} },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.memref.store`. No results.
pub fn memref_store(ctx: *mlir.Context, value: *const mlir.Value, memref: *const mlir.Value, indices: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.memref.store", .{
        .operands = .{ .flat = &.{ value, memref, indices } },
        .location = location,
    });
}

/// `fly.memref.store_vec`. No results.
pub fn memref_store_vec(ctx: *mlir.Context, vector: *const mlir.Value, memref: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.memref.store_vec", .{
        .operands = .{ .flat = &.{ vector, memref } },
        .location = location,
    });
}

/// `fly.mma.make_fragment`. Result types are inferred.
pub fn mma_make_fragment(ctx: *mlir.Context, tiled_mma: *const mlir.Value, input: *const mlir.Value, operand_id: *const mlir.Attribute, stages: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "operand_id", operand_id));
    if (stages) |present| attributes.appendAssumeCapacity(.named(ctx, "stages", present));
    return make(ctx, "fly.mma.make_fragment", .{
        .operands = .{ .flat = &.{ tiled_mma, input } },
        .attributes = attributes.constSlice(),
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.mma_atom_call`. No results.
pub fn mma_atom_call(ctx: *mlir.Context, mma_atom: *const mlir.Value, d: *const mlir.Value, a: []const *const mlir.Value, b: []const *const mlir.Value, c: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.mma_atom_call", .{
        .operands = .{ .variadic = &.{
            &.{mma_atom},
            &.{d},
            a,
            b,
            &.{c},
        } },
        .location = location,
    });
}

/// `fly.mma_atom_call_ssa`. Result types are explicit.
pub fn mma_atom_call_ssa(ctx: *mlir.Context, mma_atom: *const mlir.Value, d: ?*const mlir.Value, a: []const *const mlir.Value, b: []const *const mlir.Value, c: *const mlir.Value, results_types: []const *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.mma_atom_call_ssa", .{
        .operands = .{ .variadic = &.{
            &.{mma_atom},
            if (d) |present| &.{present} else &.{},
            a,
            b,
            &.{c},
        } },
        .results = .{ .flat = results_types },
        .location = location,
    });
}

/// `fly.prepend`. Result types are inferred.
pub fn prepend(ctx: *mlir.Context, tuple: *const mlir.Value, elem: *const mlir.Value, n: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (n) |present| attributes.appendAssumeCapacity(.named(ctx, "n", present));
    return make(ctx, "fly.prepend", .{
        .operands = .{ .flat = &.{ tuple, elem } },
        .attributes = attributes.constSlice(),
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.print`. No results.
pub fn print(ctx: *mlir.Context, values: []const *const mlir.Value, format: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.print", .{
        .operands = .{ .flat = values },
        .attributes = &.{
            .named(ctx, "format", format),
        },
        .location = location,
    });
}

/// `fly.ptr.load`. Result types are explicit.
pub fn ptr_load(ctx: *mlir.Context, ptr: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.ptr.load", .{
        .operands = .{ .flat = &.{ptr} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `fly.ptr.store`. No results.
pub fn ptr_store(ctx: *mlir.Context, value: *const mlir.Value, ptr: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.ptr.store", .{
        .operands = .{ .flat = &.{ value, ptr } },
        .location = location,
    });
}

/// `fly.ptrtoint`. Result types are inferred.
pub fn ptrtoint(ctx: *mlir.Context, ptr: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.ptrtoint", .{
        .operands = .{ .flat = &.{ptr} },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.raked_product`. Result types are inferred.
pub fn raked_product(ctx: *mlir.Context, layout: *const mlir.Value, tile: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.raked_product", .{
        .operands = .{ .flat = &.{ layout, tile } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.recast_iter`. Result types are explicit.
pub fn recast_iter(ctx: *mlir.Context, src: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.recast_iter", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `fly.recast_layout`. Result types are inferred.
pub fn recast_layout(ctx: *mlir.Context, src: *const mlir.Value, new_type_bits: *const mlir.Attribute, old_type_bits: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.recast_layout", .{
        .operands = .{ .flat = &.{src} },
        .attributes = &.{
            .named(ctx, "new_type_bits", new_type_bits),
            .named(ctx, "old_type_bits", old_type_bits),
        },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.right_inverse`. Result types are inferred.
pub fn right_inverse(ctx: *mlir.Context, layout: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.right_inverse", .{
        .operands = .{ .flat = &.{layout} },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.select`. Result types are inferred.
pub fn select(ctx: *mlir.Context, tuple: *const mlir.Value, indices: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.select", .{
        .operands = .{ .flat = &.{tuple} },
        .attributes = &.{
            .named(ctx, "indices", indices),
        },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.shape_div`. Result types are inferred.
pub fn shape_div(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.shape_div", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.size`. Result types are inferred.
pub fn size(ctx: *mlir.Context, int_tuple: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.size", .{
        .operands = .{ .flat = &.{int_tuple} },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.slice`. Result types are inferred.
pub fn slice(ctx: *mlir.Context, src: *const mlir.Value, coord: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.slice", .{
        .operands = .{ .flat = &.{ src, coord } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.static`. Result types are explicit.
pub fn static(ctx: *mlir.Context, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.static", .{
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `fly.take`. Result types are inferred.
pub fn take(ctx: *mlir.Context, tuple: *const mlir.Value, begin: *const mlir.Attribute, end: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.take", .{
        .operands = .{ .flat = &.{tuple} },
        .attributes = &.{
            .named(ctx, "begin", begin),
            .named(ctx, "end", end),
        },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.tile_to_shape`. Result types are inferred.
pub fn tile_to_shape(ctx: *mlir.Context, block: *const mlir.Value, trg_shape: *const mlir.Value, ord_shape: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.tile_to_shape", .{
        .operands = .{ .flat = &.{ block, trg_shape, ord_shape } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.tiled_copy.partition_dst`. Result types are inferred.
pub fn tiled_copy_partition_dst(ctx: *mlir.Context, tiled_copy: *const mlir.Value, dst: *const mlir.Value, coord: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.tiled_copy.partition_dst", .{
        .operands = .{ .flat = &.{ tiled_copy, dst, coord } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.tiled_copy.partition_src`. Result types are inferred.
pub fn tiled_copy_partition_src(ctx: *mlir.Context, tiled_copy: *const mlir.Value, src: *const mlir.Value, coord: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.tiled_copy.partition_src", .{
        .operands = .{ .flat = &.{ tiled_copy, src, coord } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.tiled_copy.retile`. Result types are inferred.
pub fn tiled_copy_retile(ctx: *mlir.Context, tiled_copy: *const mlir.Value, input: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.tiled_copy.retile", .{
        .operands = .{ .flat = &.{ tiled_copy, input } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.tiled_divide`. Result types are inferred.
pub fn tiled_divide(ctx: *mlir.Context, layout: *const mlir.Value, divisor: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.tiled_divide", .{
        .operands = .{ .flat = &.{ layout, divisor } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.tiled_mma.partition`. Result types are inferred.
pub fn tiled_mma_partition(ctx: *mlir.Context, tiled_mma: *const mlir.Value, input: *const mlir.Value, coord: *const mlir.Value, operand_id: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.tiled_mma.partition", .{
        .operands = .{ .flat = &.{ tiled_mma, input, coord } },
        .attributes = &.{
            .named(ctx, "operand_id", operand_id),
        },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.tiled_mma.partition_shape`. Result types are inferred.
pub fn tiled_mma_partition_shape(ctx: *mlir.Context, tiled_mma: *const mlir.Value, shape: *const mlir.Value, operand_id: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.tiled_mma.partition_shape", .{
        .operands = .{ .flat = &.{ tiled_mma, shape } },
        .attributes = &.{
            .named(ctx, "operand_id", operand_id),
        },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.tiled_product`. Result types are inferred.
pub fn tiled_product(ctx: *mlir.Context, layout: *const mlir.Value, tile: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.tiled_product", .{
        .operands = .{ .flat = &.{ layout, tile } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.to_llvm_ptr`. Result types are inferred.
pub fn to_llvm_ptr(ctx: *mlir.Context, ptr: *const mlir.Value, llvm_address_space: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.to_llvm_ptr", .{
        .operands = .{ .flat = &.{ptr} },
        .attributes = &.{
            .named(ctx, "llvm_address_space", llvm_address_space),
        },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.zipped_divide`. Result types are inferred.
pub fn zipped_divide(ctx: *mlir.Context, layout: *const mlir.Value, divisor: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.zipped_divide", .{
        .operands = .{ .flat = &.{ layout, divisor } },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly.zipped_product`. Result types are inferred.
pub fn zipped_product(ctx: *mlir.Context, layout: *const mlir.Value, tile: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly.zipped_product", .{
        .operands = .{ .flat = &.{ layout, tile } },
        .result_type_inference = true,
        .location = location,
    });
}

test {
    std.testing.refAllDecls(@This());
}
