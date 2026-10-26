const std = @import("std");

const air = @import("mlir/dialects/air");
const arith = @import("mlir/dialects/arith");
const func = @import("mlir/dialects/func");
const llvm = @import("mlir/dialects/llvm");
const mlir = @import("mlir");

const operation_names: []const []const u8 = &.{
    "air.abs.s",
    "air.ceil",
    "air.clamp",
    "air.clamp.s",
    "air.clamp.u",
    "air.convert",
    "air.convert.s",
    "air.convert.u",
    "air.cos",
    "air.dot",
    "air.exp",
    "air.exp2",
    "air.fabs",
    "air.fast_ceil",
    "air.fast_clamp",
    "air.fast_cos",
    "air.fast_exp",
    "air.fast_exp2",
    "air.fast_fabs",
    "air.fast_floor",
    "air.fast_fmax",
    "air.fast_fmin",
    "air.fast_fmod",
    "air.fast_log",
    "air.fast_log2",
    "air.fast_pow",
    "air.fast_round",
    "air.fast_rsqrt",
    "air.fast_sin",
    "air.fast_sqrt",
    "air.fast_tanh",
    "air.fast_trunc",
    "air.floor",
    "air.fma",
    "air.fmax",
    "air.fmin",
    "air.fmod",
    "air.log",
    "air.log2",
    "air.max.s",
    "air.max.u",
    "air.min.s",
    "air.min.u",
    "air.pow",
    "air.round",
    "air.rsqrt",
    "air.simd_all",
    "air.simd_any",
    "air.simd_broadcast",
    "air.simd_broadcast.s",
    "air.simd_broadcast.u",
    "air.simd_max",
    "air.simd_max.s",
    "air.simd_max.u",
    "air.simd_min",
    "air.simd_min.s",
    "air.simd_min.u",
    "air.simd_prefix_exclusive_sum",
    "air.simd_shuffle",
    "air.simd_shuffle.s",
    "air.simd_shuffle.u",
    "air.simd_shuffle_down",
    "air.simd_shuffle_down.s",
    "air.simd_shuffle_down.u",
    "air.simd_shuffle_xor",
    "air.simd_shuffle_xor.s",
    "air.simd_shuffle_xor.u",
    "air.simd_sum",
    "air.simd_sum.s",
    "air.simd_sum.u",
    "air.simdgroup.barrier",
    "air.simdgroup_matrix_8x8_load",
    "air.simdgroup_matrix_8x8_multiply_accumulate",
    "air.simdgroup_matrix_8x8_store",
    "air.sin",
    "air.sqrt",
    "air.tanh",
    "air.trunc",
    "air.wg.barrier",
};

test {
    std.testing.refAllDecls(air);
    inline for (.{ air.BuiltinAttr, air.MemFlagsAttr, air.StructFieldAttr, air.BufferAttr }) |T| std.testing.refAllDecls(T);
}

fn testContext() !*mlir.Context {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    inline for (.{ air, arith, func, llvm }) |dialect| dialect.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();
    return ctx;
}

fn expectRoundTrip(ctx: *mlir.Context, value: anytype) !void {
    var writer: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer writer.deinit();
    try writer.writer.print("{f}", .{value});
    try std.testing.expect((try mlir.Attribute.parse(ctx, writer.written())).eql(value.attribute()));
}

test "every bound operation is registered" {
    const ctx = try testContext();
    defer ctx.deinit();
    for (operation_names) |name| try std.testing.expect(ctx.isRegisteredOperation(name));
    try std.testing.expect(!ctx.allowUnregisteredDialects());
}

test "AIR attributes use the C API" {
    const ctx = try testContext();
    defer ctx.deinit();
    inline for (std.meta.tags(air.Builtin)) |builtin| {
        const attr = try air.BuiltinAttr.get(ctx, .{ .value = builtin });
        try std.testing.expectEqual(builtin, attr.value());
        try expectRoundTrip(ctx, attr);
    }
    for ([_]air.MemFlags{ .none, .{ .device = true }, .{ .threadgroup = true }, .{ .device = true, .threadgroup = true, .texture = true } }) |flags| {
        const attr = try air.MemFlagsAttr.get(ctx, .{ .value = flags });
        try std.testing.expectEqual(flags, attr.value());
        try expectRoundTrip(ctx, attr);
    }
    try std.testing.expectError(error.InvalidMlir, air.MemFlagsAttr.get(ctx, .{ .value = .{ ._padding = 1 } }));

    const field = try air.StructFieldAttr.get(ctx, .{ .offset = 4, .size = 4, .count = 3, .typeName = "float", .name = "s" });
    try std.testing.expectEqual(4, field.offset());
    try std.testing.expectEqual(4, field.size());
    try std.testing.expectEqual(3, field.count());
    try std.testing.expectEqualStrings("float", field.typeName());
    try std.testing.expectEqualStrings("s", field.name());
    try expectRoundTrip(ctx, field);
    try std.testing.expectError(error.InvalidMlir, air.StructFieldAttr.get(ctx, .{ .offset = 0, .size = 0, .typeName = "int", .name = "n" }));

    const i32_type: *const mlir.Type = .int(ctx, .i32);
    const f32_type: *const mlir.Type = .float(ctx, .f32);
    const device: air.BufferAttr.InitArgs = .{
        .locationIndex = 1,
        .addressSpace = .device,
        .access = .read,
        .elementType = .vector(&.{4}, .float(ctx, .bf16)),
        .typeName = "bfloat4",
        .typeSize = 8,
        .typeAlignment = 8,
        .noalias_ = true,
    };
    const buffer = try air.BufferAttr.get(ctx, device);
    try std.testing.expectEqual(1, buffer.locationIndex());
    try std.testing.expectEqual(air.AddressSpace.device, buffer.addressSpace());
    try std.testing.expectEqual(air.Access.read, buffer.access());
    try std.testing.expect(buffer.elementType().eql(device.elementType));
    try std.testing.expectEqualStrings("bfloat4", buffer.typeName());
    try std.testing.expectEqual(8, buffer.typeSize());
    try std.testing.expectEqual(8, buffer.typeAlignment());
    try std.testing.expectEqual(0, buffer.bufferSize());
    try std.testing.expect(buffer.noalias_());
    try std.testing.expectEqual(0, buffer.numFields());
    try expectRoundTrip(ctx, buffer);

    // `struct A { int n; float s[3]; }`, as Apple's compiler describes it.
    const fields = [_]*const air.StructFieldAttr{
        try air.StructFieldAttr.get(ctx, .{ .offset = 0, .size = 4, .typeName = "int", .name = "n" }),
        field,
    };
    const constant: air.BufferAttr.InitArgs = .{
        .locationIndex = 0,
        .addressSpace = .constant,
        .access = .read,
        .elementType = llvm.structType(ctx, &.{ i32_type, llvm.arrayType(f32_type, 3) }),
        .typeName = "A",
        .typeSize = 16,
        .typeAlignment = 4,
        .bufferSize = 16,
        .fields = &fields,
    };
    const structured = try air.BufferAttr.get(ctx, constant);
    try std.testing.expectEqual(2, structured.numFields());
    try std.testing.expect(structured.field(1).eql(field));
    try expectRoundTrip(ctx, structured);

    var invalid = device;
    invalid.typeSize = 16;
    try std.testing.expectError(error.InvalidMlir, air.BufferAttr.get(ctx, invalid));
    invalid = device;
    invalid.addressSpace = .constant;
    invalid.access = .read_write;
    try std.testing.expectError(error.InvalidMlir, air.BufferAttr.get(ctx, invalid));
    invalid = device;
    invalid.addressSpace = .thread;
    try std.testing.expectError(error.InvalidMlir, air.BufferAttr.get(ctx, invalid));
    invalid = device;
    invalid.addressSpace = .threadgroup;
    try std.testing.expectError(error.InvalidMlir, air.BufferAttr.get(ctx, invalid));
    invalid = device;
    invalid.bufferSize = 8;
    try std.testing.expectError(error.InvalidMlir, air.BufferAttr.get(ctx, invalid));
    invalid = device;
    invalid.fields = fields[0..1];
    try std.testing.expectError(error.InvalidMlir, air.BufferAttr.get(ctx, invalid));
    invalid = constant;
    invalid.fields = fields[0..1];
    try std.testing.expectError(error.InvalidMlir, air.BufferAttr.get(ctx, invalid));
    invalid = constant;
    invalid.addressSpace = .device;
    try std.testing.expectError(error.InvalidMlir, air.BufferAttr.get(ctx, invalid));

    const other = try testContext();
    defer other.deinit();
    try std.testing.expectError(error.InvalidMlir, air.BufferAttr.get(other, device));
}

test "all AIR operations construct and verify" {
    const ctx = try testContext();
    defer ctx.deinit();
    const loc: *const mlir.Location = .unknown(ctx);
    const f32x4: *const mlir.Type = .vector(&.{4}, .float(ctx, .f32));
    const i32x2: *const mlir.Type = .vector(&.{2}, .int(ctx, .i32));
    const i64x2: *const mlir.Type = .vector(&.{2}, .int(ctx, .i64));
    const block = mlir.Block.init(
        &.{ .float(ctx, .f32), .float(ctx, .f16), f32x4, .int(ctx, .i32), i32x2, .int(ctx, .i16), .int(ctx, .i1), llvm.pointerType(ctx, air.AddressSpace.threadgroup), i64x2, .vector(&.{64}, .float(ctx, .f16)), .vector(&.{64}, .float(ctx, .f32)), .vector(&.{4}, .float(ctx, .bf16)) },
        &.{ loc, loc, loc, loc, loc, loc, loc, loc, loc, loc, loc, loc },
    );
    defer block.deinit();
    const f = block.argument(0);
    const h = block.argument(1);
    const v = block.argument(2);
    const i = block.argument(3);
    const iv = block.argument(4);
    const lane = block.argument(5);
    const b = block.argument(6);
    const fast: air.FloatOpts = .{ .fastmath = .fast };
    var ops: std.ArrayList(*mlir.Operation) = .empty;
    defer ops.deinit(std.testing.allocator);
    inline for (.{ air.exp, air.exp2, air.log, air.log2, air.tanh, air.sqrt, air.rsqrt, air.fabs, air.ceil, air.floor, air.round, air.trunc, air.sin, air.cos, air.simd_sum, air.simd_max, air.simd_min, air.simd_prefix_exclusive_sum }) |op| {
        try ops.append(std.testing.allocator, op(ctx, h, .{}, loc).appendTo(block));
        try ops.append(std.testing.allocator, op(ctx, v, fast, loc).appendTo(block));
    }
    inline for (.{ air.fast_exp, air.fast_exp2, air.fast_log, air.fast_log2, air.fast_tanh, air.fast_sqrt, air.fast_rsqrt, air.fast_fabs, air.fast_ceil, air.fast_floor, air.fast_round, air.fast_trunc, air.fast_sin, air.fast_cos }) |op| {
        try ops.append(std.testing.allocator, op(ctx, f, fast, loc).appendTo(block));
        try ops.append(std.testing.allocator, op(ctx, v, fast, loc).appendTo(block));
    }
    inline for (.{ air.fmax, air.fmin, air.pow, air.fmod, air.fast_fmax, air.fast_fmin, air.fast_pow, air.fast_fmod }) |op| {
        try ops.append(std.testing.allocator, op(ctx, f, f, fast, loc).appendTo(block));
    }
    inline for (.{ air.fma, air.clamp, air.fast_clamp }) |op| {
        try ops.append(std.testing.allocator, op(ctx, v, v, v, fast, loc).appendTo(block));
    }
    try ops.append(std.testing.allocator, air.dot(ctx, v, v, fast, loc).appendTo(block));
    try ops.append(std.testing.allocator, air.abs_s(ctx, i, loc).appendTo(block));
    inline for (.{ air.max_s, air.max_u, air.min_s, air.min_u }) |op| {
        try ops.append(std.testing.allocator, op(ctx, iv, iv, loc).appendTo(block));
    }
    inline for (.{ air.clamp_s, air.clamp_u }) |op| {
        try ops.append(std.testing.allocator, op(ctx, i, i, i, loc).appendTo(block));
    }
    inline for (.{ air.simd_sum_s, air.simd_sum_u, air.simd_max_s, air.simd_max_u, air.simd_min_s, air.simd_min_u }) |op| {
        try ops.append(std.testing.allocator, op(ctx, i, loc).appendTo(block));
        try ops.append(std.testing.allocator, op(ctx, lane, loc).appendTo(block));
    }
    inline for (.{ air.simd_shuffle, air.simd_shuffle_xor, air.simd_shuffle_down, air.simd_broadcast }) |op| {
        try ops.append(std.testing.allocator, op(ctx, v, lane, fast, loc).appendTo(block));
    }
    inline for (.{ air.simd_shuffle_s, air.simd_shuffle_u, air.simd_shuffle_xor_s, air.simd_shuffle_xor_u, air.simd_shuffle_down_s, air.simd_shuffle_down_u, air.simd_broadcast_s, air.simd_broadcast_u }) |op| {
        try ops.append(std.testing.allocator, op(ctx, iv, lane, loc).appendTo(block));
    }
    inline for (.{ air.simd_any, air.simd_all }) |op| {
        try ops.append(std.testing.allocator, op(ctx, b, loc).appendTo(block));
    }
    try ops.append(std.testing.allocator, air.convert(ctx, block.argument(11), f32x4, fast, loc).appendTo(block));
    try ops.append(std.testing.allocator, air.convert(ctx, h, .float(ctx, .bf16), .{}, loc).appendTo(block));
    try ops.append(std.testing.allocator, air.convert_s(ctx, i, .float(ctx, .f32), fast, loc).appendTo(block));
    try ops.append(std.testing.allocator, air.convert_u(ctx, f, .int(ctx, .i32), .{}, loc).appendTo(block));
    inline for (.{ air.wg_barrier, air.simdgroup_barrier }) |op| {
        try ops.append(std.testing.allocator, op(ctx, .{ .threadgroup = true }, loc).appendTo(block));
    }
    const shape = block.argument(8);
    const matrix = air.simdgroup_matrix_8x8_load(ctx, block.argument(7), shape, shape, shape, .vector(&.{64}, .float(ctx, .f16)), fast, loc).appendTo(block);
    const product = air.simdgroup_matrix_8x8_multiply_accumulate(ctx, matrix.result(0), block.argument(9), block.argument(10), .vector(&.{64}, .float(ctx, .f32)), fast, loc).appendTo(block);
    const stored = air.simdgroup_matrix_8x8_store(ctx, product.result(0), block.argument(7), shape, shape, shape, loc).appendTo(block);
    try ops.appendSlice(std.testing.allocator, &.{ matrix, product, stored });
    for (ops.items) |op| try std.testing.expect(op.verify());
}

test "AIR operations round trip through their custom assembly" {
    const ctx = try testContext();
    defer ctx.deinit();
    const source =
        \\func.func @ops(%f: f32, %v: vector<4xbf16>, %i: i32, %s: i16, %b: i1, %p: !llvm.ptr<3>, %d: vector<2xi64>, %m: vector<64xf32>) {
        \\  %0 = air.fast_exp %f fastmath<fast> : f32
        \\  %1 = air.tanh %f : f32
        \\  %2 = air.fma %f, %f, %f fastmath<fast> : f32
        \\  %3 = air.max.s %i, %i : i32
        \\  %4 = air.simd_shuffle_xor %f, %s fastmath<fast> : f32
        \\  %5 = air.simd_sum.u %i : i32
        \\  %6 = air.simd_any %b
        \\  %7 = air.convert %v fastmath<fast> : vector<4xbf16> to vector<4xf32>
        \\  %8 = air.convert.s %i fastmath<fast> : i32 to f32
        \\  %9 = air.dot %7, %7 fastmath<fast> : vector<4xf32> -> f32
        \\  air.wg.barrier <threadgroup>
        \\  air.simdgroup.barrier <device|threadgroup>
        \\  %10 = air.simdgroup_matrix_8x8_load %p, %d, %d, %d fastmath<fast> : !llvm.ptr<3> -> vector<64xf32>
        \\  %11 = air.simdgroup_matrix_8x8_multiply_accumulate %10, %m, %m fastmath<fast> : vector<64xf32>, vector<64xf32>, vector<64xf32> -> vector<64xf32>
        \\  air.simdgroup_matrix_8x8_store %11, %p, %d, %d, %d : vector<64xf32>, !llvm.ptr<3>
        \\  return
        \\}
    ;
    const module = try mlir.Module.parse(ctx, source);
    defer module.deinit();
    var first: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer first.deinit();
    try first.writer.print("{f}", .{module.operation()});
    try std.testing.expect(std.mem.find(u8, first.written(), "\"air.") == null);
    const reparsed = try mlir.Module.parse(ctx, first.written());
    defer reparsed.deinit();
    var second: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer second.deinit();
    try second.writer.print("{f}", .{reparsed.operation()});
    try std.testing.expectEqualStrings(first.written(), second.written());
}

test "AIR verifier rejects types Apple's compiler has no intrinsic for" {
    const ctx = try testContext();
    defer ctx.deinit();
    const rejected = [_][]const u8{
        "func.func @f(%x: f16) { %0 = air.fast_exp %x : f16\n return }",
        "func.func @f(%x: bf16) { %0 = air.exp %x : bf16\n return }",
        "func.func @f(%x: f32) { %0 = air.max.s %x, %x : f32\n return }",
        "func.func @f(%x: i64) { %0 = air.simd_sum.s %x : i64\n return }",
        "func.func @f(%x: vector<8xf32>) { %0 = air.simd_sum %x : vector<8xf32>\n return }",
        "func.func @f(%x: f32) { %0 = air.convert %x : f32 to f32\n return }",
        "func.func @f(%x: i32) { %0 = air.convert %x : i32 to f32\n return }",
        "func.func @f(%x: f32) { %0 = air.convert.s %x : f32 to f16\n return }",
        "func.func @f(%x: f32) { %0 = air.convert.u %x fastmath<fast> : f32 to i32\n return }",
        "func.func @f(%x: vector<4xf32>) { %0 = air.convert %x : vector<4xf32> to vector<2xf16>\n return }",
        "func.func @f(%x: vector<4xf32>) { %0 = air.dot %x, %x : vector<4xf32> -> f16\n return }",
        "func.func @f(%p: !llvm.ptr<2>, %d: vector<2xi64>) { %0 = air.simdgroup_matrix_8x8_load %p, %d, %d, %d : !llvm.ptr<2> -> vector<64xf32>\n return }",
        "func.func @f(%a: vector<64xbf16>, %b: vector<64xf16>, %c: vector<64xf32>) { %0 = air.simdgroup_matrix_8x8_multiply_accumulate %a, %b, %c : vector<64xbf16>, vector<64xf16>, vector<64xf32> -> vector<64xf32>\n return }",
        "func.func @f(%a: vector<64xf16>, %c: vector<64xf32>) { %0 = air.simdgroup_matrix_8x8_multiply_accumulate %a, %a, %c : vector<64xf16>, vector<64xf16>, vector<64xf32> -> vector<64xf16>\n return }",
    };
    for (rejected) |source| try std.testing.expectError(error.InvalidMlir, mlir.Module.parse(ctx, source));
}

test "AIR kernel ABI" {
    const ctx = try testContext();
    defer ctx.deinit();
    const device = "#air.buffer<locationIndex = 0, addressSpace = device, access = read_write, elementType = f32, typeName = \"float\", typeSize = 4, typeAlignment = 4>";
    const threadgroup = "#air.buffer<locationIndex = 0, addressSpace = threadgroup, access = read_write, elementType = f16, typeName = \"half\", typeSize = 2, typeAlignment = 2>";
    const accepted = [_][]const u8{
        "func.func @k(%a: !llvm.ptr<1> {air.buffer = " ++ device ++ ", air.arg_name = \"a\"}, %b: !llvm.ptr<3> {air.buffer = " ++ threadgroup ++ "}, %t: vector<3xi32> {air.builtin = #air.builtin<threadgroup_position_in_grid>}, %l: i16 {air.builtin = #air.builtin<thread_index_in_simdgroup>}, %s: i32 {air.builtin = #air.builtin<simdgroup_index_in_threadgroup>}) attributes {air.kernel} { return }",
        "module attributes {air.fast_math_enable} {}",
    };
    const rejected = [_][]const u8{
        "func.func @k(%a: i32) attributes {air.kernel} { return }",
        "func.func @k(%a: !llvm.ptr<1> {air.buffer = " ++ device ++ ", air.builtin = #air.builtin<thread_position_in_grid>}) attributes {air.kernel} { return }",
        "func.func @k(%a: !llvm.ptr<3> {air.buffer = " ++ device ++ "}) attributes {air.kernel} { return }",
        "func.func @k(%a: !llvm.ptr<1> {air.buffer = " ++ device ++ "}, %b: !llvm.ptr<1> {air.buffer = " ++ device ++ "}) attributes {air.kernel} { return }",
        "func.func @k(%a: vector<3xi32> {air.builtin = #air.builtin<thread_index_in_simdgroup>}) attributes {air.kernel} { return }",
        "func.func @k(%a: vector<4xi32> {air.builtin = #air.builtin<thread_position_in_grid>}) attributes {air.kernel} { return }",
        "func.func @k(%a: i64 {air.builtin = #air.builtin<thread_position_in_grid>}) attributes {air.kernel} { return }",
        "func.func @k(%a: i32 {air.builtin = #air.builtin<thread_position_in_grid>}, %b: i32 {air.builtin = #air.builtin<thread_position_in_grid>}) attributes {air.kernel} { return }",
        "func.func @k(%a: i32 {air.builtin = #air.builtin<thread_position_in_grid>, air.arg_name = \"\"}) attributes {air.kernel} { return }",
        "func.func @k() -> i32 attributes {air.kernel} { %c = arith.constant 0 : i32\n return %c : i32 }",
        "func.func @h(%a: i32 {air.builtin = #air.builtin<thread_position_in_grid>}) { return }",
        "func.func @h() attributes {air.fast_math_enable} { return }",
        "func.func @h() attributes {air.unknown} { return }",
    };
    for (accepted) |source| {
        const module = try mlir.Module.parse(ctx, source);
        module.deinit();
    }
    for (rejected) |source| try std.testing.expectError(error.InvalidMlir, mlir.Module.parse(ctx, source));
}

test "AIR kernel builder" {
    const ctx = try testContext();
    defer ctx.deinit();
    const loc: *const mlir.Location = .unknown(ctx);
    const module: *mlir.Module = .init(loc);
    defer module.deinit();
    const block = mlir.Block.init(&.{ llvm.pointerType(ctx, air.AddressSpace.device), .vector(&.{3}, .int(ctx, .i32)) }, &.{ loc, loc });
    _ = air.wg_barrier(ctx, .{ .threadgroup = true }, loc).appendTo(block);
    _ = func.returns(ctx, &.{}, loc).appendTo(block);
    _ = air.kernel(ctx, .{
        .name = "example",
        .block = block,
        .location = loc,
        .args_attributes = &.{
            try air.bufferArgumentAttr(ctx, .{
                .locationIndex = 0,
                .addressSpace = .device,
                .access = .read_write,
                .elementType = .float(ctx, .f32),
                .typeName = "float",
                .typeSize = 4,
                .typeAlignment = 4,
            }, .{ .name = "out" }),
            try air.builtinArgumentAttr(ctx, .thread_position_in_grid, .{}),
        },
    }).appendTo(module.body());
    air.enableFastMath(ctx, module.operation());
    try std.testing.expect(module.operation().verify());
    var writer: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer writer.deinit();
    try writer.writer.print("{f}", .{module.operation()});
    const parsed = try mlir.Module.parse(ctx, writer.written());
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
    try std.testing.expect(std.mem.find(u8, writer.written(), "air.arg_name = \"out\"") != null);
}
