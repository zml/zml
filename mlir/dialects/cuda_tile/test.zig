const std = @import("std");

const c = @import("c");
const cuda_tile = @import("mlir/dialects/cuda_tile");
const mlir = @import("mlir");

fn testContext() !*mlir.Context {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    cuda_tile.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();
    return ctx;
}

test {
    std.testing.refAllDecls(cuda_tile);
}

test "dialect registers and types round-trip" {
    const ctx = try testContext();
    defer ctx.deinit();

    try std.testing.expect(ctx.isRegisteredOperation("cuda_tile.entry"));
    try std.testing.expect(ctx.isRegisteredOperation("cuda_tile.load_view_tko"));
    try std.testing.expect(ctx.isRegisteredOperation("cuda_tile.mmaf"));

    const f32_ty = mlir.Type.float(ctx, .f32);
    const tile = cuda_tile.tileType(ctx, &.{ 64, 64 }, f32_ty);
    try std.testing.expect(tile.eql(try mlir.Type.parse(ctx, "!cuda_tile.tile<64x64xf32>")));
    try std.testing.expectEqual(@as(usize, 2), tile.isA(cuda_tile.TileType).?.rank());
    try std.testing.expectEqual(@as(i64, 64), tile.isA(cuda_tile.TileType).?.dimension(1));

    const ptr = cuda_tile.pointerType(ctx, f32_ty);
    try std.testing.expect(ptr.eql(try mlir.Type.parse(ctx, "!cuda_tile.ptr<f32>")));
    try std.testing.expect(ptr.isA(cuda_tile.PointerType).?.pointee().eql(f32_ty));

    const dyn = cuda_tile.TensorViewType.dynamic();
    const tv = cuda_tile.tensorViewType(ctx, f32_ty, &.{ dyn, 32 }, &.{ dyn, 1 });
    try std.testing.expect(tv.eql(try mlir.Type.parse(ctx, "!cuda_tile.tensor_view<?x32xf32, strides=[?,1]>")));

    const static_tv = cuda_tile.tensorViewType(ctx, f32_ty, &.{16}, &.{1});
    const pv: *const mlir.Type = @ptrCast(cuda_tile.PartitionViewType.get(ctx, .{ .tile_shape = &.{2}, .tensor_view = static_tv, .padding = .zero }));
    try std.testing.expect(pv.eql(try mlir.Type.parse(ctx, "!cuda_tile.partition_view<tile=(2), padding_value = zero, tensor_view<16xf32, strides=[1]>>")));
    try std.testing.expect(cuda_tile.tileTypeOfView(pv).eql(cuda_tile.tileType(ctx, &.{2}, f32_ty)));
    try std.testing.expectEqual(@as(usize, 1), cuda_tile.indexRankOfView(pv));

    try std.testing.expect(cuda_tile.tokenType(ctx).eql(try mlir.Type.parse(ctx, "!cuda_tile.token")));

    const gather_tv = cuda_tile.tensorViewType(ctx, f32_ty, &.{ 64, 32 }, &.{ 32, 1 });
    const gsv = cuda_tile.gatherScatterViewType(ctx, &.{ 8, 16 }, gather_tv, 0, .{ .padding = .zero });
    try std.testing.expect(gsv.eql(try mlir.Type.parse(ctx, "!cuda_tile.gather_scatter_view<tile=(8x16), padding_value = zero, tensor_view<64x32xf32, strides=[32,1]>, sparse_dim=0>")));
    const typed_gsv = gsv.isA(cuda_tile.GatherScatterViewType).?;
    const gsv_tile = typed_gsv.getTileShape();
    var gsv_shape: [2]i32 = undefined;
    try std.testing.expectEqual(gsv_shape.len, gsv_tile.numElements());
    for (&gsv_shape, 0..) |*d, i| d.* = gsv_tile.element(i);
    const rebuilt_gsv = try cuda_tile.GatherScatterViewType.get(ctx, .{
        .tileShape = &gsv_shape,
        .tensorView = typed_gsv.getTensorView(),
        .sparseDim = typed_gsv.getSparseDim(),
        .paddingValue = typed_gsv.getPaddingValue(),
    });
    try std.testing.expect(rebuilt_gsv.eql(typed_gsv));
    try std.testing.expect(cuda_tile.tileTypeOfView(gsv).eql(cuda_tile.tileType(ctx, &.{ 8, 16 }, f32_ty)));
    try std.testing.expect((try cuda_tile.GatherScatterViewType.get(ctx, .{
        .tileShape = &.{ 8, 16 },
        .tensorView = gather_tv.isA(cuda_tile.TensorViewType).?,
        .sparseDim = 1,
    })).getPaddingValue() == null);
    try std.testing.expectError(error.InvalidMlir, cuda_tile.GatherScatterViewType.get(ctx, .{
        .tileShape = &.{8},
        .tensorView = gather_tv.isA(cuda_tile.TensorViewType).?,
        .sparseDim = 0,
    }));
}

test "enum and hint attributes" {
    const ctx = try testContext();
    defer ctx.deinit();

    try std.testing.expect(cuda_tile.RoundingMode.nearest_even.attribute(ctx).eql(try mlir.Attribute.parse(ctx, "#cuda_tile.rounding<nearest_even>")));
    const padding = cuda_tile.PaddingValue.neg_inf.attribute(ctx);
    const padding_text = c.mlirCudaTilePaddingValueAttrGetValue(padding.ptr());
    try std.testing.expectEqualStrings("neg_inf", padding_text.data[0..padding_text.length]);

    const hints = cuda_tile.optimizationHintsAttr(ctx, &[_]cuda_tile.EntryHint{
        .{ .arch = .sm_100, .num_cta_in_cga = 2 },
        .{ .arch = .sm_120, .num_cta_in_cga = 2, .occupancy = 2 },
    });
    try std.testing.expect(hints.eql(try mlir.Attribute.parse(ctx, "#cuda_tile.optimization_hints<sm_100 = {num_cta_in_cga = 2}, sm_120 = {num_cta_in_cga = 2, occupancy = 2}>")));
    const typed_hints = hints.isA(cuda_tile.OptimizationHintsAttr).?;
    try std.testing.expect((try cuda_tile.OptimizationHintsAttr.get(ctx, .{ .value = typed_hints.getValue() })).eql(typed_hints));
    const ls_hints = cuda_tile.optimizationHintsAttr(ctx, &[_]cuda_tile.LoadStoreHint{.{ .arch = .sm_100, .allow_tma = false, .latency = 3 }});
    try std.testing.expect(ls_hints.eql(try mlir.Attribute.parse(ctx, "#cuda_tile.optimization_hints<sm_100 = {allow_tma = false, latency = 3}>")));
    const not_a_dict: *const mlir.DictionaryAttribute = @ptrCast(mlir.Attribute.dict(ctx, &.{.named(ctx, "sm_100", .int(ctx, .i64, 1))}));
    try std.testing.expectError(error.InvalidMlir, cuda_tile.OptimizationHintsAttr.get(ctx, .{ .value = not_a_dict }));

    const div = cuda_tile.divByAttr(ctx, 16, .{});
    try std.testing.expect(div.eql(try mlir.Attribute.parse(ctx, "#cuda_tile.div_by<16>")));
    const div_every = cuda_tile.divByAttr(ctx, 4, .{ .every = 4, .along = 1 });
    try std.testing.expect(div_every.eql(try mlir.Attribute.parse(ctx, "#cuda_tile.div_by<4, every 4 along 1>")));
    try std.testing.expect(cuda_tile.divByAttr(ctx, 32, .{ .every = 4 }).eql(try mlir.Attribute.parse(ctx, "#cuda_tile.div_by<32, every 4 along 0>")));
    const typed_div = div_every.isA(cuda_tile.DivByAttr).?;
    try std.testing.expect((try cuda_tile.DivByAttr.get(ctx, .{
        .divisor = typed_div.getDivisor(),
        .every = typed_div.getEvery(),
        .along = typed_div.getAlong(),
    })).eql(typed_div));
    try std.testing.expect(div.isA(cuda_tile.DivByAttr).?.getEvery() == null);
    try std.testing.expectError(error.InvalidMlir, cuda_tile.DivByAttr.get(ctx, .{ .divisor = 4, .every = 4 }));

    const same = cuda_tile.sameElementsAttr(ctx, &.{ 2, 4 });
    try std.testing.expect(same.eql(try mlir.Attribute.parse(ctx, "#cuda_tile.same_elements<[2, 4]>")));
    const same_values = same.isA(cuda_tile.SameElementsAttr).?.getValues();
    try std.testing.expectEqual(@as(usize, 2), same_values.numElements());
    try std.testing.expectEqual(@as(i64, 4), same_values.element(1));

    const lower = cuda_tile.boundedAttr(ctx, .{ .lb = 0 });
    try std.testing.expect(lower.eql(try mlir.Attribute.parse(ctx, "#cuda_tile.bounded<0, ?>")));
    try std.testing.expect(cuda_tile.boundedAttr(ctx, .{ .ub = 7 }).eql(try mlir.Attribute.parse(ctx, "#cuda_tile.bounded<?, 7>")));
    try std.testing.expect(cuda_tile.boundedAttr(ctx, .{ .lb = -3, .ub = 7 }).eql(try mlir.Attribute.parse(ctx, "#cuda_tile.bounded<-3, 7>")));
    const typed_lower = lower.isA(cuda_tile.BoundedAttr).?;
    try std.testing.expectEqual(@as(?i64, 0), typed_lower.getLb());
    try std.testing.expect(typed_lower.getUb() == null);
}

test "entry round-trip through print and parse" {
    const ctx = try testContext();
    defer ctx.deinit();

    const loc: *const mlir.Location = .unknown(ctx);
    const module_op: *mlir.Module = .init(loc);
    defer module_op.deinit();

    const f32_ty = mlir.Type.float(ctx, .f32);
    const ptr_tile = cuda_tile.tileType(ctx, &.{}, cuda_tile.pointerType(ctx, f32_ty));
    const block = mlir.Block.init(&.{ ptr_tile, ptr_tile }, &.{ loc, loc });

    // out[i] = in[i] + 1 over 128 elements, the shape of cuda-tile's README example.
    const idx_ty = cuda_tile.tileType(ctx, &.{128}, .int(ctx, .i32));
    const offsets = cuda_tile.iota(ctx, idx_ty, loc).appendTo(block);
    const ptr1 = cuda_tile.tileType(ctx, &.{1}, cuda_tile.pointerType(ctx, f32_ty));
    const ptr128 = cuda_tile.tileType(ctx, &.{128}, cuda_tile.pointerType(ctx, f32_ty));
    const in_r = cuda_tile.reshape(ctx, block.argument(0), ptr1, loc).appendTo(block);
    const in_b = cuda_tile.broadcast(ctx, in_r.result(0), ptr128, loc).appendTo(block);
    const in_p = cuda_tile.offset(ctx, in_b.result(0), offsets.result(0), loc).appendTo(block);
    const f32_128 = cuda_tile.tileType(ctx, &.{128}, f32_ty);
    const loaded = cuda_tile.load_ptr_tko(ctx, in_p.result(0), f32_128, .{}, loc).appendTo(block);
    const one = cuda_tile.constant(ctx, cuda_tile.splatAttr(f32_128, cuda_tile.floatElemAttr(ctx, f32_ty, 1.0)), f32_128, loc).appendTo(block);
    const sum = cuda_tile.addf(ctx, loaded.result(0), one.result(0), .nearest_even, false, loc).appendTo(block);
    const out_r = cuda_tile.reshape(ctx, block.argument(1), ptr1, loc).appendTo(block);
    const out_b = cuda_tile.broadcast(ctx, out_r.result(0), ptr128, loc).appendTo(block);
    const out_p = cuda_tile.offset(ctx, out_b.result(0), offsets.result(0), loc).appendTo(block);
    _ = cuda_tile.store_ptr_tko(ctx, out_p.result(0), sum.result(0), .{ .mem = .{ .token = loaded.result(1) } }, loc).appendTo(block);
    _ = cuda_tile.return_(ctx, &.{}, loc).appendTo(block);

    const ct_body = mlir.Block.init(&.{}, &.{});
    _ = cuda_tile.entry(ctx, .{ .name = "add_one", .block = block, .location = loc }).appendTo(ct_body);
    _ = cuda_tile.module(ctx, "m", ct_body, loc).appendTo(module_op.body());

    try std.testing.expect(module_op.operation().verify());

    var al: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer al.deinit();
    try al.writer.print("{f}", .{module_op.operation()});
    const text = al.written();
    try std.testing.expect(std.mem.indexOf(u8, text, "cuda_tile.module @m") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "entry @add_one(") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "load_ptr_tko weak") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "addf") != null);

    const parsed = try mlir.Module.parse(ctx, text);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}

test "views, control flow and mma verify" {
    const ctx = try testContext();
    defer ctx.deinit();

    const loc: *const mlir.Location = .unknown(ctx);
    const module_op: *mlir.Module = .init(loc);
    defer module_op.deinit();

    const bf16_ty = mlir.Type.float(ctx, .bf16);
    const f32_ty = mlir.Type.float(ctx, .f32);
    const i32_ty = mlir.Type.int(ctx, .i32);
    const bf16_ptr = cuda_tile.tileType(ctx, &.{}, cuda_tile.pointerType(ctx, bf16_ty));
    const f32_ptr = cuda_tile.tileType(ctx, &.{}, cuda_tile.pointerType(ctx, f32_ty));
    const block = mlir.Block.init(&.{ bf16_ptr, bf16_ptr, f32_ptr }, &.{ loc, loc, loc });

    // C[128x128] = A[128x64] @ B[64x128] in 64-wide K tiles.
    const tv_a = cuda_tile.make_tensor_view(ctx, block.argument(0), &.{}, &.{}, cuda_tile.tensorViewType(ctx, bf16_ty, &.{ 128, 64 }, &.{ 64, 1 }), loc).appendTo(block);
    const tv_b = cuda_tile.make_tensor_view(ctx, block.argument(1), &.{}, &.{}, cuda_tile.tensorViewType(ctx, bf16_ty, &.{ 64, 128 }, &.{ 128, 1 }), loc).appendTo(block);
    const tv_c = cuda_tile.make_tensor_view(ctx, block.argument(2), &.{}, &.{}, cuda_tile.tensorViewType(ctx, f32_ty, &.{ 128, 128 }, &.{ 128, 1 }), loc).appendTo(block);
    const pv_a: *const mlir.Type = @ptrCast(cuda_tile.PartitionViewType.get(ctx, .{ .tile_shape = &.{ 64, 64 }, .tensor_view = tv_a.result(0).type_(), .padding = .zero }));
    const pv_b: *const mlir.Type = @ptrCast(cuda_tile.PartitionViewType.get(ctx, .{ .tile_shape = &.{ 64, 64 }, .tensor_view = tv_b.result(0).type_(), .padding = .zero }));
    const pv_c: *const mlir.Type = @ptrCast(cuda_tile.PartitionViewType.get(ctx, .{ .tile_shape = &.{ 64, 64 }, .tensor_view = tv_c.result(0).type_() }));
    const va = cuda_tile.make_partition_view(ctx, tv_a.result(0), pv_a, loc).appendTo(block);
    const vb = cuda_tile.make_partition_view(ctx, tv_b.result(0), pv_b, loc).appendTo(block);
    const vc = cuda_tile.make_partition_view(ctx, tv_c.result(0), pv_c, loc).appendTo(block);

    const bid = cuda_tile.get_tile_block_id(ctx, loc).appendTo(block);
    const i32_tile = cuda_tile.tileType(ctx, &.{}, i32_ty);
    const zero = cuda_tile.constant(ctx, cuda_tile.splatAttr(i32_tile, cuda_tile.intElemAttr(i32_ty, 0)), i32_tile, loc).appendTo(block);
    const one = cuda_tile.constant(ctx, cuda_tile.splatAttr(i32_tile, cuda_tile.intElemAttr(i32_ty, 1)), i32_tile, loc).appendTo(block);
    const acc_ty = cuda_tile.tileType(ctx, &.{ 64, 64 }, f32_ty);
    const acc0 = cuda_tile.constant(ctx, cuda_tile.splatAttr(acc_ty, cuda_tile.floatElemAttr(ctx, f32_ty, 0.0)), acc_ty, loc).appendTo(block);

    const body = mlir.Block.init(&.{ i32_tile, acc_ty }, &.{ loc, loc });
    const at = cuda_tile.load_view_tko(ctx, va.result(0), &.{ bid.result(0), body.argument(0) }, .{}, loc).appendTo(body);
    const bt = cuda_tile.load_view_tko(ctx, vb.result(0), &.{ body.argument(0), bid.result(1) }, .{}, loc).appendTo(body);
    try std.testing.expect(at.attributeByName("inbounds").?.eql(.denseArray(ctx, .bool, &.{ 0, 0 })));
    const acc = cuda_tile.mmaf(ctx, at.result(0), bt.result(0), body.argument(1), false, loc).appendTo(body);
    _ = cuda_tile.continue_(ctx, &.{acc.result(0)}, loc).appendTo(body);
    const loop_op = cuda_tile.for_(ctx, zero.result(0), one.result(0), one.result(0), &.{acc0.result(0)}, body, false, loc).appendTo(block);

    // if %cond -> (tile<f32>) { yield } else { yield }
    const cond = cuda_tile.cmpi(ctx, .equal, .signed, bid.result(0), zero.result(0), loc).appendTo(block);
    const then_block = mlir.Block.init(&.{}, &.{});
    _ = cuda_tile.yield(ctx, &.{loop_op.result(0)}, loc).appendTo(then_block);
    const else_block = mlir.Block.init(&.{}, &.{});
    _ = cuda_tile.yield(ctx, &.{acc0.result(0)}, loc).appendTo(else_block);
    const if_op = cuda_tile.if_(ctx, cond.result(0), &.{acc_ty}, then_block, .{ .else_block = else_block }, loc).appendTo(block);

    const store = cuda_tile.store_view_tko(ctx, if_op.result(0), vc.result(0), &.{ bid.result(0), bid.result(1) }, .{}, loc).appendTo(block);
    try std.testing.expect(store.attributeByName("inbounds").?.eql(.denseArray(ctx, .bool, &.{ 0, 0 })));
    _ = cuda_tile.return_(ctx, &.{}, loc).appendTo(block);

    const ct_body = mlir.Block.init(&.{}, &.{});
    _ = cuda_tile.entry(ctx, .{
        .name = "gemm",
        .block = block,
        .optimization_hints = cuda_tile.optimizationHintsAttr(ctx, &[_]cuda_tile.EntryHint{.{ .arch = .sm_120, .num_worker_warps_per_cta = 4 }}),
        .location = loc,
    }).appendTo(ct_body);
    _ = cuda_tile.module(ctx, "m", ct_body, loc).appendTo(module_op.body());

    try std.testing.expect(module_op.operation().verify());

    var al: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer al.deinit();
    try al.writer.print("{f}", .{module_op.operation()});
    const text = al.written();
    try std.testing.expect(std.mem.indexOf(u8, text, "optimization_hints=<sm_120 = {num_worker_warps_per_cta = 4}>") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "load_view_tko weak") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "continue ") != null);

    const parsed = try mlir.Module.parse(ctx, text);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}

test "the verifier rejects what the dialect forbids" {
    const ctx = try testContext();
    defer ctx.deinit();
    const loc: *const mlir.Location = .unknown(ctx);
    const f32_ty = mlir.Type.float(ctx, .f32);

    // An entry with a rank-1 argument. (Per-op negatives go through
    // `Operation.try_make`, whose logged error the test runner counts as a
    // failure, so the module-level verifier is the oracle here.)
    const module_op: *mlir.Module = .init(loc);
    defer module_op.deinit();
    const block = mlir.Block.init(&.{cuda_tile.tileType(ctx, &.{4}, f32_ty)}, &.{loc});
    _ = cuda_tile.return_(ctx, &.{}, loc).appendTo(block);
    const ct_body = mlir.Block.init(&.{}, &.{});
    _ = cuda_tile.entry(ctx, .{ .name = "bad", .block = block, .location = loc }).appendTo(ct_body);
    _ = cuda_tile.module(ctx, "m", ct_body, loc).appendTo(module_op.body());
    try std.testing.expect(!module_op.operation().verify());
}

test "every bound operation is registered" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    cuda_tile.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    for (cuda_tile.operation_names) |name| try std.testing.expect(ctx.isRegisteredOperation(name));
    try std.testing.expect(!ctx.allowUnregisteredDialects());
}
