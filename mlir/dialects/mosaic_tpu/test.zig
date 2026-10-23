const std = @import("std");

const mlir = @import("mlir");
const mosaic_tpu = @import("mlir/dialects/mosaic_tpu");

fn testContext() !*mlir.Context {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    mosaic_tpu.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();
    return ctx;
}

fn expectSegments(op: *const mlir.Operation, expected: []const i32) !void {
    const segments = op.attributeByName("operandSegmentSizes").?.isA(mlir.DenseArrayAttribute(.i32)).?;
    try std.testing.expectEqual(expected.len, segments.numElements());
    for (expected, 0..) |size, i| try std.testing.expectEqual(size, segments.element(i));
}

fn expectPrints(expected: []const u8, value: anytype) !void {
    var buf: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try w.print("{f}", .{value});
    try std.testing.expectEqualStrings(expected, w.buffered());
}

fn expectEnumAttr(comptime Attr: type, ctx: *mlir.Context) !void {
    const Enum = @FieldType(Attr.InitArgs, "value");
    inline for (@typeInfo(Enum).@"enum".field_names) |name| {
        const value: Enum = @field(Enum, name);
        const attr = try Attr.get(ctx, .{ .value = value });
        try std.testing.expectEqual(value, attr.value());
        try std.testing.expect(attr.attribute().isA(Attr) != null);
        try std.testing.expect(value.attribute(ctx).eql(attr.attribute()));
        try expectPrints("#tpu." ++ Attr.mnemonic ++ "<" ++ name ++ ">", attr);
    }
}

test {
    @setEvalBranchQuota(100_000);
    std.testing.refAllDecls(mosaic_tpu);
    inline for (comptime std.meta.declarations(mosaic_tpu)) |decl| {
        const T = @field(mosaic_tpu, decl);
        if (@TypeOf(T) == type and @typeInfo(T) == .@"opaque") std.testing.refAllDecls(T);
    }
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

    const dma = mosaic_tpu.enqueue_dma(ctx, source, target, dma_sem, .{}, loc);
    defer dma.deinit();
    try expectSegments(dma, &.{ 1, 0, 1, 1, 0, 0, 0 });
    try std.testing.expect(dma.verify());
    const signal = mosaic_tpu.sem_signal(ctx, sem, amount, .{}, loc);
    defer signal.deinit();
    try expectSegments(signal, &.{ 1, 1, 0, 0, 0 });
    try std.testing.expect(signal.verify());

    const sc_sem = args.addArgument(try mlir.Type.parse(ctx, "memref<!tpu.semaphore, #tpu.memory_space<semaphore_mem, sc_vector_subcore>>"), loc);
    const sc_dma_sem = args.addArgument(try mlir.Type.parse(ctx, "memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem, sc_vector_subcore>>"), loc);
    const core = args.addArgument(.int(ctx, .i32), loc);
    const subcore = args.addArgument(.int(ctx, .i32), loc);
    const remote_signal = mosaic_tpu.sem_signal(ctx, sc_sem, amount, .{ .core_id = core, .subcore_id = subcore }, loc);
    defer remote_signal.deinit();
    try expectSegments(remote_signal, &.{ 1, 1, 0, 1, 1 });
    try std.testing.expect(remote_signal.operand(3).eql(subcore));
    const remote_dma = mosaic_tpu.enqueue_dma(ctx, source, target, sc_dma_sem, .{
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
    const product = mosaic_tpu.matmul(ctx, lhs, rhs, acc, vector_type, .{}, loc);
    defer product.deinit();
    try std.testing.expect(product.verify());
    try std.testing.expect(!product.attributeByName("transpose_lhs_hint").?.isA(mlir.BoolAttribute).?.value());

    const ref_type = try mlir.Type.parse(ctx, "memref<8x8xf32, #tpu.memory_space<vmem>>");
    const ref = args.addArgument(ref_type, loc);
    const cast = mosaic_tpu.reinterpret_cast(ctx, ref, ref_type, loc);
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
    const float_min = mosaic_tpu.all_reduce(ctx, floats, 1, .min, float_type, loc);
    defer float_min.deinit();
    const int_max = mosaic_tpu.all_reduce(ctx, integers, 1, .max, int_type, loc);
    defer int_max.deinit();
    try std.testing.expect(float_min.attributeByName("kind").?.eql(try mlir.Attribute.parse(ctx, "#tpu.reduction_kind<minf>")));
    try std.testing.expect(int_max.attributeByName("kind").?.eql(try mlir.Attribute.parse(ctx, "#tpu.reduction_kind<maxsi>")));

    const int_scan = mosaic_tpu.scan(ctx, integers, .max, int_type, .{ .mask = mask }, loc);
    defer int_scan.deinit();
    try std.testing.expect(int_scan.verify());
    try std.testing.expect(int_scan.attributeByName("kind").?.eql(try mlir.Attribute.parse(ctx, "#tpu.reduction_kind<maxui>")));
    try std.testing.expectEqual(@as(i64, 1), int_scan.attributeByName("dimension").?.isA(mlir.IntegerAttribute).?.value(i64));
    const float_scan = mosaic_tpu.scan(ctx, floats, .min, float_type, .{}, loc);
    defer float_scan.deinit();
    try std.testing.expect(float_scan.attributeByName("kind").?.eql(try mlir.Attribute.parse(ctx, "#tpu.reduction_kind<minf>")));
}

test "tpu types are built through the C API" {
    const ctx = try mlir.Context.init(.{});
    defer ctx.deinit();

    const semaphore = try mosaic_tpu.SemaphoreType.get(ctx, .{});
    try std.testing.expect(mosaic_tpu.semaphoreType(ctx).isA(mosaic_tpu.SemaphoreType) != null);
    try std.testing.expect(mosaic_tpu.semaphoreType(ctx).isA(mosaic_tpu.DMASemaphoreType) == null);
    try expectPrints("!tpu.semaphore", semaphore);

    const dma_semaphore = try mosaic_tpu.DMASemaphoreType.get(ctx, .{});
    try std.testing.expect(mosaic_tpu.dmaSemaphoreType(ctx).isA(mosaic_tpu.DMASemaphoreType) != null);
    try expectPrints("!tpu.dma_semaphore", dma_semaphore);
}

test "tpu enum attributes are built through the C API" {
    const ctx = try mlir.Context.init(.{});
    defer ctx.deinit();

    try expectEnumAttr(mosaic_tpu.ReductionKindAttr, ctx);
    try expectEnumAttr(mosaic_tpu.ContractPrecisionAttr, ctx);
    try expectEnumAttr(mosaic_tpu.RoundingModeAttr, ctx);
    try expectEnumAttr(mosaic_tpu.CoreTypeAttr, ctx);
    try expectEnumAttr(mosaic_tpu.DimensionSemanticsAttr, ctx);
    try expectEnumAttr(mosaic_tpu.PipelineModeAttr, ctx);
    try expectEnumAttr(mosaic_tpu.RevisitModeAttr, ctx);
    try expectEnumAttr(mosaic_tpu.MemorySpaceAttr, ctx);

    const smem_tc = try mosaic_tpu.MemorySpaceAttr.get(ctx, .{ .value = .smem, .coreType = .sc_scalar_subcore });
    try std.testing.expectEqual(mosaic_tpu.MemorySpace.smem, smem_tc.value());
    try std.testing.expectEqual(mosaic_tpu.CoreType.sc_scalar_subcore, smem_tc.coreType().?);
    try expectPrints("#tpu.memory_space<smem, sc_scalar_subcore>", smem_tc);
    try std.testing.expectEqual(null, (try mosaic_tpu.MemorySpaceAttr.get(ctx, .{ .value = .hbm })).coreType());
    // The dialect verifier's pairing rules: no core type on hbm, no scalar subcore on vmem.
    try std.testing.expectError(error.InvalidMlir, mosaic_tpu.MemorySpaceAttr.get(ctx, .{ .value = .hbm, .coreType = .tc }));
    try std.testing.expectError(error.InvalidMlir, mosaic_tpu.MemorySpaceAttr.get(ctx, .{ .value = .vmem, .coreType = .sc_scalar_subcore }));
}

test "tpu array attributes are built through the C API" {
    const ctx = try mlir.Context.init(.{});
    defer ctx.deinit();

    const window = try mosaic_tpu.ElementWindowAttr.get(ctx, .{ .padLow = &.{ 0, 2 }, .padHigh = &.{ 1, 3 } });
    try std.testing.expectEqual(2, window.numPadLow());
    try std.testing.expectEqual(2, window.padLow(1));
    try std.testing.expectEqual(2, window.numPadHigh());
    try std.testing.expectEqual(3, window.padHigh(1));
    try expectPrints("#tpu.element_window<[0, 2], [1, 3]>", window);
    try std.testing.expect(mosaic_tpu.elementWindowAttr(ctx, &.{ 0, 2 }, &.{ 1, 3 }).eql(window.attribute()));

    const dims = mosaic_tpu.dotDimensionNumbersAttr(ctx, &.{1}, &.{0}, &.{0}, &.{1}, &.{ 0, 0, 1, 1 }, &.{}, &.{});
    try expectPrints("#tpu.dot_dimension_numbers<[1], [0], [0], [1], [0, 0, 1, 1], [], []>", dims);

    const batched = try mosaic_tpu.DotDimensionNumbersAttr.get(ctx, .{
        .lhsContractingDims = &.{2},
        .rhsContractingDims = &.{1},
        .lhsNonContractingDims = &.{1},
        .outputDimOrder = &.{ 0, 0, 0, 1 },
        .lhsBatchDims = &.{0},
        .rhsBatchDims = &.{0},
    });
    try std.testing.expectEqual(0, batched.numRhsNonContractingDims());
    try std.testing.expectEqual(4, batched.numOutputDimOrder());
    try std.testing.expectEqual(1, batched.outputDimOrder(3));
    try std.testing.expectEqual(0, batched.rhsBatchDims(0));
    try expectPrints("#tpu.dot_dimension_numbers<[2], [1], [1], [], [0, 0, 0, 1], [0], [0]>", batched);
}

test "every bound operation is registered" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    mosaic_tpu.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    for (mosaic_tpu.operation_names) |name| try std.testing.expect(ctx.isRegisteredOperation(name));
    try std.testing.expect(!ctx.allowUnregisteredDialects());
}
