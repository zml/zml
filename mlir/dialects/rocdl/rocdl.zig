//! LLVM ROCDL operations, AMDGPU attributes, and GPU target configuration.
//! All operation builders from the pinned LLVM revision are available in `ops`.

const std = @import("std");
const c = @import("c");

const mlir = @import("mlir");

pub const types = @import("types.zig");
pub const annotations = @import("annotations.zig");
pub const AddressSpace = types.AddressSpace;
pub const pointerType = types.pointerType;

pub const ops = @import("ops.zig");
pub const enums = @import("enums.zig");
pub const attributes = @import("attributes.zig");

pub const BufferOOBModeAttr = attributes.BufferOOBModeAttr;
pub const BufferOOBModeModuleFlagAttr = attributes.BufferOOBModeModuleFlagAttr;
pub const Gfx12AtomicCachePolicyAttr = attributes.Gfx12AtomicCachePolicyAttr;
pub const Gfx12CachePolicyAttr = attributes.Gfx12CachePolicyAttr;
pub const Gfx942CachePolicyAttr = attributes.Gfx942CachePolicyAttr;
pub const MFMANegModifierAttr = attributes.MFMANegModifierAttr;
pub const MFMAPermBAttr = attributes.MFMAPermBAttr;
pub const MatrixFormatAttr = attributes.MatrixFormatAttr;
pub const PreGfx12CachePolicyAttr = attributes.PreGfx12CachePolicyAttr;
pub const SchedGroupMaskAttr = attributes.SchedGroupMaskAttr;
pub const TBufferOOBModeModuleFlagAttr = attributes.TBufferOOBModeModuleFlagAttr;
pub const TargetAttr = attributes.TargetAttr;
pub const WMMACModifierAttr = attributes.WMMACModifierAttr;
pub const WMMAMatrixScaleAttr = attributes.WMMAMatrixScaleAttr;
pub const WMMAMatrixScaleFormatAttr = attributes.WMMAMatrixScaleFormatAttr;

pub const BufferOOBMode = enums.BufferOOBMode;
pub const Gfx12AtomicCachePolicy = enums.Gfx12AtomicCachePolicy;
pub const Gfx12CachePolicy = enums.Gfx12CachePolicy;
pub const Gfx942CachePolicy = enums.Gfx942CachePolicy;
pub const MFMANegModifier = enums.MFMANegModifier;
pub const MFMAPermB = enums.MFMAPermB;
pub const MatrixFormat = enums.MatrixFormat;
pub const PreGfx12CachePolicy = enums.PreGfx12CachePolicy;
pub const SchedGroupMask = enums.SchedGroupMask;
pub const WMMACModifier = enums.WMMACModifier;
pub const WMMAMatrixScale = enums.WMMAMatrixScale;
pub const WMMAMatrixScaleFormat = enums.WMMAMatrixScaleFormat;

pub fn registerDialect(registry: *mlir.DialectRegistry) void {
    registry.registerDialect("rocdl");
}

/// Converts two packed FP8 values from the selected word to vector<2xf32>.
pub fn cvtPkF32Fp8(
    ctx: *mlir.Context,
    bits: *const mlir.Value,
    high: bool,
    location: *const mlir.Location,
) *mlir.Operation {
    return ops.cvt_pk_f32_fp8(ctx, bits, .vector(&.{2}, .float(ctx, .f32)), .boolean(ctx, high), location);
}

test {
    @setEvalBranchQuota(100_000);
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(ops);
    std.testing.refAllDecls(types);
    std.testing.refAllDecls(annotations);
    inline for (attributes.enum_attributes) |A| std.testing.refAllDecls(A);
    std.testing.refAllDecls(TargetAttr);
}

test "packed FP8 conversion uses the registered LLVM ROCDL dialect" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    try std.testing.expect(ctx.isRegisteredOperation("rocdl.cvt.pk.f32.fp8"));

    const loc = mlir.Location.unknown(ctx);
    const block = mlir.Block.init(&.{.int(ctx, .i32)}, &.{loc});
    defer block.deinit();
    for ([_]bool{ false, true }) |high| {
        const op = cvtPkF32Fp8(ctx, block.argument(0), high, loc);
        defer op.deinit();
        try std.testing.expect(op.verify());
        try std.testing.expect(op.result(0).type_().eql(.vector(&.{2}, .float(ctx, .f32))));
        try std.testing.expect(op.attributeByName("wordSel").?.eql(.boolean(ctx, high)));
    }
}

fn testContext() !*mlir.Context {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry });
    ctx.loadAllAvailableDialects();
    return ctx;
}

test "every registered ROCDL operation has a generated builder" {
    const ctx = try testContext();
    defer ctx.deinit();
    try std.testing.expectEqual(ops.names.len, @as(usize, @intCast(c.mlirROCDLGetOperationCount(ctx.ptr()))));
    for (ops.names, 0..) |name, i| {
        try std.testing.expect(ctx.isRegisteredOperation(name));
        for (ops.names[0..i]) |earlier| try std.testing.expect(!std.mem.eql(u8, name, earlier));
    }
}

fn expectAttributeRoundTrip(ctx: *mlir.Context, attribute: *const mlir.Attribute) !void {
    var buffer: [2048]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buffer);
    try attribute.format(&writer);
    try std.testing.expect(attribute.eql(try mlir.Attribute.parse(ctx, writer.buffered())));
}

test "all ROCDL enum attributes construct and round-trip every named value" {
    @setEvalBranchQuota(100_000);
    const ctx = try testContext();
    defer ctx.deinit();
    inline for (attributes.enum_attributes) |A| {
        for (std.enums.values(A.Value)) |value| {
            const attribute = try A.get(ctx, value);
            try std.testing.expect(attribute.isA(A) != null);
            try std.testing.expectEqual(value, attribute.value());
            try expectAttributeRoundTrip(ctx, attribute.attribute());
        }
    }
    const mask: SchedGroupMask = .combine(&.{ .valu, .mfma_wmma, .ds_read });
    try std.testing.expect(mask.contains(.mfma_wmma));
    try std.testing.expect(!mask.contains(.all_vmem));
    const attribute = try SchedGroupMaskAttr.get(ctx, mask);
    try std.testing.expectEqual(mask, attribute.value());
    try expectAttributeRoundTrip(ctx, attribute.attribute());
}

test "ROCDL enum constructors reject invalid values and flag bits" {
    const ctx = try testContext();
    defer ctx.deinit();
    try std.testing.expect(c.mlirROCDLMatrixFormatAttrGet(ctx.ptr(), 5).ptr == null);
    try std.testing.expect(c.mlirROCDLWMMACModifierAttrGet(ctx.ptr(), 65536).ptr == null);
    try std.testing.expectError(error.InvalidMlir, SchedGroupMaskAttr.get(ctx, @enumFromInt(1 << 20)));
    try std.testing.expectError(error.InvalidMlir, PreGfx12CachePolicyAttr.get(ctx, @enumFromInt(1 << 6)));
}

test "target attributes expose all fields and reject invalid options" {
    const ctx = try testContext();
    defer ctx.deinit();
    const defaults = try TargetAttr.get(ctx, .{});
    try std.testing.expectEqual(@as(i32, 2), defaults.optLevel());
    try std.testing.expectEqualStrings("amdgcn-amd-amdhsa", defaults.triple());
    try std.testing.expectEqualStrings("gfx900", defaults.chip());
    try std.testing.expectEqualStrings("", defaults.features());
    try std.testing.expectEqualStrings("600", defaults.abi());
    try std.testing.expect(defaults.flags() == null and defaults.link() == null);
    try std.testing.expect(defaults.hasWave64());
    try std.testing.expect(!defaults.hasFastMath());
    try std.testing.expect(defaults.hasCorrectSqrt());
    try expectAttributeRoundTrip(ctx, defaults.attribute());

    const flags: *const mlir.Attribute = @ptrCast(mlir.DictionaryAttribute.init(ctx, &.{
        .named(ctx, "no_wave64", .unit(ctx)),
        .named(ctx, "fast", .unit(ctx)),
        .named(ctx, "daz", .unit(ctx)),
        .named(ctx, "finite_only", .unit(ctx)),
        .named(ctx, "unsafe_math", .unit(ctx)),
        .named(ctx, "unsafe_sqrt", .unit(ctx)),
    }));
    const link: *const mlir.Attribute = @ptrCast(mlir.ArrayAttribute.init(ctx, &.{.string(ctx, "device.bc")}));
    const target = try TargetAttr.get(ctx, .{ .optLevel = 3, .chip = "gfx942", .features = "+xnack", .abi = "500", .flags = flags, .link = link });
    try std.testing.expectEqual(@as(i32, 3), target.optLevel());
    try std.testing.expectEqualStrings("gfx942", target.chip());
    try std.testing.expectEqualStrings("+xnack", target.features());
    try std.testing.expectEqualStrings("500", target.abi());
    try std.testing.expect(target.flags().?.eql(flags));
    try std.testing.expect(target.link().?.eql(link));
    try std.testing.expect(target.hasFlag("fast") and target.hasFastMath());
    try std.testing.expect(!target.hasWave64());
    try std.testing.expect(target.hasDaz() and target.hasFiniteOnly());
    try std.testing.expect(target.hasUnsafeMath() and !target.hasCorrectSqrt());
    try expectAttributeRoundTrip(ctx, target.attribute());

    try std.testing.expectError(error.InvalidMlir, TargetAttr.get(ctx, .{ .optLevel = 4 }));
    try std.testing.expectError(error.InvalidMlir, TargetAttr.get(ctx, .{ .triple = "" }));
    try std.testing.expectError(error.InvalidMlir, TargetAttr.get(ctx, .{ .chip = "" }));
    try std.testing.expectError(error.InvalidMlir, TargetAttr.get(ctx, .{ .abi = "700" }));
    try std.testing.expectError(error.InvalidMlir, TargetAttr.get(ctx, .{ .flags = .unit(ctx) }));
    const badLink: *const mlir.Attribute = @ptrCast(mlir.ArrayAttribute.init(ctx, &.{.unit(ctx)}));
    try std.testing.expectError(error.InvalidMlir, TargetAttr.get(ctx, .{ .link = badLink }));
    const other = try testContext();
    defer other.deinit();
    const otherFlags: *const mlir.Attribute = @ptrCast(mlir.DictionaryAttribute.init(other, &.{}));
    try std.testing.expectError(error.InvalidMlir, TargetAttr.get(ctx, .{ .flags = otherFlags }));
}

test "ROCDL barriers buffer loads and matrix operations verify and reparse" {
    const ctx = try testContext();
    defer ctx.deinit();
    const loc = mlir.Location.unknown(ctx);
    const module = try mlir.Module.parse(ctx,
        \\module {
        \\  llvm.func @test(%rsrc: vector<4xi32>, %offset: i32, %a: f32, %c: vector<4xf32>) {
        \\    llvm.return
        \\  }
        \\}
    );
    defer module.deinit();
    const function = module.body().firstOperation().?;
    const block = function.region(0).firstBlock().?;
    const terminator = block.terminator().?;
    const i32Type: *const mlir.Type = .int(ctx, .i32);
    const f32Type: *const mlir.Type = .float(ctx, .f32);
    const vec4f32: *const mlir.Type = .vector(&.{4}, f32Type);
    const zero: *const mlir.Attribute = .int(ctx, .i32, 0);
    const opList = .{
        ops.s_barrier(ctx, loc),
        ops.s_sleep(ctx, .int(ctx, .i32, 1), loc),
        ops.workitem_id_x(ctx, i32Type, .{}, loc),
        ops.sched_barrier(ctx, (try SchedGroupMaskAttr.get(ctx, .combine(&.{ .valu, .mfma_wmma }))).attribute(), loc),
        ops.raw_buffer_load(ctx, block.argument(0), block.argument(1), block.argument(1), f32Type, .{}, loc),
        ops.raw_buffer_load(
            ctx,
            block.argument(0),
            block.argument(1),
            block.argument(1),
            f32Type,
            .{ .aux = (try Gfx942CachePolicyAttr.get(ctx, .combine(&.{ .sc0, .nt }))).attribute() },
            loc,
        ),
        ops.mfma_f32_16x16x4f32(ctx, block.argument(2), block.argument(2), block.argument(3), vec4f32, zero, zero, (try MFMAPermBAttr.get(ctx, .none)).attribute(), loc),
    };
    inline for (opList) |op| {
        try std.testing.expect(op.verify());
        block.insertOwnedOperationBefore(terminator, op);
    }
    annotations.annotateFunction(ctx, function, .{
        .kernel = true,
        .reqdWorkGroupSize = .{ 64, 1, 1 },
        .flatWorkGroupSize = "64,64",
        .maxFlatWorkGroupSize = 64,
        .wavesPerEu = 2,
        .unsafeFpAtomics = true,
    });
    annotations.annotateMemory(ctx, opList[4], .{ .lastUse = true, .noRemoteMemory = true, .noFineGrainedMemory = true, .ignoreDenormalMode = true });
    try std.testing.expect(opList[4].attributeByName("aux").?.eql(.int(ctx, .i32, 0)));
    try std.testing.expect(opList[5].attributeByName("aux").?.isA(Gfx942CachePolicyAttr) != null);
    try std.testing.expect(module.operation().verify());
    var buffer: [16384]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buffer);
    try module.format(&writer);
    const parsed = try mlir.Module.parse(ctx, writer.buffered());
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}

test "ROCDL annotations preserve module flag defaults and explicit false" {
    const ctx = try testContext();
    defer ctx.deinit();
    const module = mlir.Module.init(.unknown(ctx));
    defer module.deinit();
    annotations.annotateModule(ctx, module.operation(), .{});
    try std.testing.expect(module.operation().attributeByName("rocdl.xnack") == null);
    annotations.annotateModule(ctx, module.operation(), .{ .xnack = false, .sramecc = true });
    try std.testing.expect(module.operation().attributeByName("rocdl.xnack").?.eql(.boolean(ctx, false)));
    try std.testing.expect(module.operation().attributeByName("rocdl.sramecc").?.eql(.boolean(ctx, true)));
    try std.testing.expect(module.operation().verify());
    const pointer = pointerType(ctx, .barrier);
    try std.testing.expect(pointer.eql(try mlir.Type.parse(ctx, "!llvm.ptr<15>")));
}
