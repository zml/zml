const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");
const rocdl = @import("mlir/dialects/rocdl");

const operation_names: []const []const u8 = &.{
    "rocdl.asyncmark",
    "rocdl.ballot",
    "rocdl.barrier",
    "rocdl.cluster.id.x",
    "rocdl.cluster.id.y",
    "rocdl.cluster.id.z",
    "rocdl.cluster.load.async.to.lds.b128",
    "rocdl.cluster.load.async.to.lds.b32",
    "rocdl.cluster.load.async.to.lds.b64",
    "rocdl.cluster.load.async.to.lds.b8",
    "rocdl.cluster.workgroup.id.x",
    "rocdl.cluster.workgroup.id.y",
    "rocdl.cluster.workgroup.id.z",
    "rocdl.cos",
    "rocdl.cvt.f32.bf8",
    "rocdl.cvt.f32.fp8",
    "rocdl.cvt.pk.bf8.f32",
    "rocdl.cvt.pk.f32.bf8",
    "rocdl.cvt.pk.f32.fp8",
    "rocdl.cvt.pk.fp8.f32",
    "rocdl.cvt.pkrtz",
    "rocdl.cvt.scale.pk16.bf16.bf6",
    "rocdl.cvt.scale.pk16.bf16.fp6",
    "rocdl.cvt.scale.pk16.f16.bf6",
    "rocdl.cvt.scale.pk16.f16.fp6",
    "rocdl.cvt.scale.pk16.f32.bf6",
    "rocdl.cvt.scale.pk16.f32.fp6",
    "rocdl.cvt.scale.pk8.bf16.bf8",
    "rocdl.cvt.scale.pk8.bf16.fp4",
    "rocdl.cvt.scale.pk8.bf16.fp8",
    "rocdl.cvt.scale.pk8.f16.bf8",
    "rocdl.cvt.scale.pk8.f16.fp4",
    "rocdl.cvt.scale.pk8.f16.fp8",
    "rocdl.cvt.scale.pk8.f32.bf8",
    "rocdl.cvt.scale.pk8.f32.fp4",
    "rocdl.cvt.scale.pk8.f32.fp8",
    "rocdl.cvt.scalef32.2xpk16.bf6.f32",
    "rocdl.cvt.scalef32.2xpk16.fp6.f32",
    "rocdl.cvt.scalef32.f16.bf8",
    "rocdl.cvt.scalef32.f16.fp8",
    "rocdl.cvt.scalef32.f32.bf8",
    "rocdl.cvt.scalef32.f32.fp8",
    "rocdl.cvt.scalef32.pk.bf16.bf8",
    "rocdl.cvt.scalef32.pk.bf16.fp4",
    "rocdl.cvt.scalef32.pk.bf16.fp8",
    "rocdl.cvt.scalef32.pk.bf8.bf16",
    "rocdl.cvt.scalef32.pk.bf8.f16",
    "rocdl.cvt.scalef32.pk.bf8.f32",
    "rocdl.cvt.scalef32.pk.f16.bf8",
    "rocdl.cvt.scalef32.pk.f16.fp4",
    "rocdl.cvt.scalef32.pk.f16.fp8",
    "rocdl.cvt.scalef32.pk.f32.bf8",
    "rocdl.cvt.scalef32.pk.f32.fp4",
    "rocdl.cvt.scalef32.pk.f32.fp8",
    "rocdl.cvt.scalef32.pk.fp4.bf16",
    "rocdl.cvt.scalef32.pk.fp4.f16",
    "rocdl.cvt.scalef32.pk.fp4.f32",
    "rocdl.cvt.scalef32.pk.fp8.bf16",
    "rocdl.cvt.scalef32.pk.fp8.f16",
    "rocdl.cvt.scalef32.pk.fp8.f32",
    "rocdl.cvt.scalef32.pk16.bf6.bf16",
    "rocdl.cvt.scalef32.pk16.bf6.f16",
    "rocdl.cvt.scalef32.pk16.bf6.f32",
    "rocdl.cvt.scalef32.pk16.fp6.bf16",
    "rocdl.cvt.scalef32.pk16.fp6.f16",
    "rocdl.cvt.scalef32.pk16.fp6.f32",
    "rocdl.cvt.scalef32.pk32.bf16.bf6",
    "rocdl.cvt.scalef32.pk32.bf16.fp6",
    "rocdl.cvt.scalef32.pk32.bf6.bf16",
    "rocdl.cvt.scalef32.pk32.bf6.f16",
    "rocdl.cvt.scalef32.pk32.f16.bf6",
    "rocdl.cvt.scalef32.pk32.f16.fp6",
    "rocdl.cvt.scalef32.pk32.f32.bf6",
    "rocdl.cvt.scalef32.pk32.f32.fp6",
    "rocdl.cvt.scalef32.pk32.fp6.bf16",
    "rocdl.cvt.scalef32.pk32.fp6.f16",
    "rocdl.cvt.scalef32.pk8.bf8.bf16",
    "rocdl.cvt.scalef32.pk8.bf8.f16",
    "rocdl.cvt.scalef32.pk8.bf8.f32",
    "rocdl.cvt.scalef32.pk8.fp4.bf16",
    "rocdl.cvt.scalef32.pk8.fp4.f16",
    "rocdl.cvt.scalef32.pk8.fp4.f32",
    "rocdl.cvt.scalef32.pk8.fp8.bf16",
    "rocdl.cvt.scalef32.pk8.fp8.f16",
    "rocdl.cvt.scalef32.pk8.fp8.f32",
    "rocdl.cvt.scalef32.sr.bf8.bf16",
    "rocdl.cvt.scalef32.sr.bf8.f16",
    "rocdl.cvt.scalef32.sr.bf8.f32",
    "rocdl.cvt.scalef32.sr.fp8.bf16",
    "rocdl.cvt.scalef32.sr.fp8.f16",
    "rocdl.cvt.scalef32.sr.fp8.f32",
    "rocdl.cvt.scalef32.sr.pk.fp4.bf16",
    "rocdl.cvt.scalef32.sr.pk.fp4.f16",
    "rocdl.cvt.scalef32.sr.pk.fp4.f32",
    "rocdl.cvt.scalef32.sr.pk16.bf6.bf16",
    "rocdl.cvt.scalef32.sr.pk16.bf6.f16",
    "rocdl.cvt.scalef32.sr.pk16.bf6.f32",
    "rocdl.cvt.scalef32.sr.pk16.fp6.bf16",
    "rocdl.cvt.scalef32.sr.pk16.fp6.f16",
    "rocdl.cvt.scalef32.sr.pk16.fp6.f32",
    "rocdl.cvt.scalef32.sr.pk32.bf6.bf16",
    "rocdl.cvt.scalef32.sr.pk32.bf6.f16",
    "rocdl.cvt.scalef32.sr.pk32.bf6.f32",
    "rocdl.cvt.scalef32.sr.pk32.fp6.bf16",
    "rocdl.cvt.scalef32.sr.pk32.fp6.f16",
    "rocdl.cvt.scalef32.sr.pk32.fp6.f32",
    "rocdl.cvt.scalef32.sr.pk8.bf8.bf16",
    "rocdl.cvt.scalef32.sr.pk8.bf8.f16",
    "rocdl.cvt.scalef32.sr.pk8.bf8.f32",
    "rocdl.cvt.scalef32.sr.pk8.fp4.bf16",
    "rocdl.cvt.scalef32.sr.pk8.fp4.f16",
    "rocdl.cvt.scalef32.sr.pk8.fp4.f32",
    "rocdl.cvt.scalef32.sr.pk8.fp8.bf16",
    "rocdl.cvt.scalef32.sr.pk8.fp8.f16",
    "rocdl.cvt.scalef32.sr.pk8.fp8.f32",
    "rocdl.cvt.sr.bf8.f32",
    "rocdl.cvt.sr.fp8.f32",
    "rocdl.dot4.f32.bf8.bf8",
    "rocdl.dot4.f32.bf8.fp8",
    "rocdl.dot4.f32.fp8.bf8",
    "rocdl.dot4.f32.fp8.fp8",
    "rocdl.ds.atomic.async.barrier.arrive.b64",
    "rocdl.ds.atomic.barrier.arrive.rtn.b64",
    "rocdl.ds.load.tr16.b128",
    "rocdl.ds.load.tr4.b64",
    "rocdl.ds.load.tr6.b96",
    "rocdl.ds.load.tr8.b64",
    "rocdl.ds.read.tr16.b64",
    "rocdl.ds.read.tr4.b64",
    "rocdl.ds.read.tr6.b96",
    "rocdl.ds.read.tr8.b64",
    "rocdl.ds_bpermute",
    "rocdl.ds_swizzle",
    "rocdl.exp",
    "rocdl.exp2",
    "rocdl.fdot2",
    "rocdl.fdot2.bf16.bf16",
    "rocdl.fdot2.f16.f16",
    "rocdl.fdot2.f32.bf16",
    "rocdl.flat.prefetch",
    "rocdl.fmed3",
    "rocdl.global.load.async.lds",
    "rocdl.global.load.async.to.lds.b128",
    "rocdl.global.load.async.to.lds.b32",
    "rocdl.global.load.async.to.lds.b64",
    "rocdl.global.load.async.to.lds.b8",
    "rocdl.global.load.lds",
    "rocdl.global.load.tr.b128",
    "rocdl.global.load.tr.b64",
    "rocdl.global.load.tr4.b64",
    "rocdl.global.load.tr6.b96",
    "rocdl.global.prefetch",
    "rocdl.global.store.async.from.lds.b128",
    "rocdl.global.store.async.from.lds.b32",
    "rocdl.global.store.async.from.lds.b64",
    "rocdl.global.store.async.from.lds.b8",
    "rocdl.iglp.opt",
    "rocdl.load.async.to.lds",
    "rocdl.load.to.lds",
    "rocdl.log",
    "rocdl.make.buffer.rsrc",
    "rocdl.mbcnt.hi",
    "rocdl.mbcnt.lo",
    "rocdl.mfma.f32.16x16x16bf16.1k",
    "rocdl.mfma.f32.16x16x16f16",
    "rocdl.mfma.f32.16x16x1f32",
    "rocdl.mfma.f32.16x16x2bf16",
    "rocdl.mfma.f32.16x16x32.bf16",
    "rocdl.mfma.f32.16x16x32.bf8.bf8",
    "rocdl.mfma.f32.16x16x32.bf8.fp8",
    "rocdl.mfma.f32.16x16x32.f16",
    "rocdl.mfma.f32.16x16x32.fp8.bf8",
    "rocdl.mfma.f32.16x16x32.fp8.fp8",
    "rocdl.mfma.f32.16x16x4bf16.1k",
    "rocdl.mfma.f32.16x16x4f16",
    "rocdl.mfma.f32.16x16x4f32",
    "rocdl.mfma.f32.16x16x8.xf32",
    "rocdl.mfma.f32.16x16x8bf16",
    "rocdl.mfma.f32.32x32x16.bf16",
    "rocdl.mfma.f32.32x32x16.bf8.bf8",
    "rocdl.mfma.f32.32x32x16.bf8.fp8",
    "rocdl.mfma.f32.32x32x16.f16",
    "rocdl.mfma.f32.32x32x16.fp8.bf8",
    "rocdl.mfma.f32.32x32x16.fp8.fp8",
    "rocdl.mfma.f32.32x32x1f32",
    "rocdl.mfma.f32.32x32x2bf16",
    "rocdl.mfma.f32.32x32x2f32",
    "rocdl.mfma.f32.32x32x4.xf32",
    "rocdl.mfma.f32.32x32x4bf16",
    "rocdl.mfma.f32.32x32x4bf16.1k",
    "rocdl.mfma.f32.32x32x4f16",
    "rocdl.mfma.f32.32x32x8bf16.1k",
    "rocdl.mfma.f32.32x32x8f16",
    "rocdl.mfma.f32.4x4x1f32",
    "rocdl.mfma.f32.4x4x2bf16",
    "rocdl.mfma.f32.4x4x4bf16.1k",
    "rocdl.mfma.f32.4x4x4f16",
    "rocdl.mfma.f64.16x16x4f64",
    "rocdl.mfma.f64.4x4x4f64",
    "rocdl.mfma.i32.16x16x16i8",
    "rocdl.mfma.i32.16x16x32.i8",
    "rocdl.mfma.i32.16x16x4i8",
    "rocdl.mfma.i32.16x16x64.i8",
    "rocdl.mfma.i32.32x32x16.i8",
    "rocdl.mfma.i32.32x32x32.i8",
    "rocdl.mfma.i32.32x32x4i8",
    "rocdl.mfma.i32.32x32x8i8",
    "rocdl.mfma.i32.4x4x4i8",
    "rocdl.mfma.scale.f32.16x16x128.f8f6f4",
    "rocdl.mfma.scale.f32.32x32x64.f8f6f4",
    "rocdl.permlane16.swap",
    "rocdl.permlane16.var",
    "rocdl.permlane32.swap",
    "rocdl.permlanex16",
    "rocdl.permlanex16.var",
    "rocdl.ptr.s.buffer.load",
    "rocdl.raw.buffer.atomic.cmpswap",
    "rocdl.raw.buffer.atomic.fadd",
    "rocdl.raw.buffer.atomic.fmax",
    "rocdl.raw.buffer.atomic.smax",
    "rocdl.raw.buffer.atomic.umin",
    "rocdl.raw.buffer.load",
    "rocdl.raw.buffer.store",
    "rocdl.raw.ptr.buffer.atomic.cmpswap",
    "rocdl.raw.ptr.buffer.atomic.fadd",
    "rocdl.raw.ptr.buffer.atomic.fmax",
    "rocdl.raw.ptr.buffer.atomic.smax",
    "rocdl.raw.ptr.buffer.atomic.umin",
    "rocdl.raw.ptr.buffer.load",
    "rocdl.raw.ptr.buffer.load.async.lds",
    "rocdl.raw.ptr.buffer.load.lds",
    "rocdl.raw.ptr.buffer.store",
    "rocdl.rcp",
    "rocdl.readfirstlane",
    "rocdl.readlane",
    "rocdl.rsq",
    "rocdl.s.barrier",
    "rocdl.s.barrier.init",
    "rocdl.s.barrier.join",
    "rocdl.s.barrier.leave",
    "rocdl.s.barrier.signal",
    "rocdl.s.barrier.signal.isfirst",
    "rocdl.s.barrier.signal.var",
    "rocdl.s.barrier.wait",
    "rocdl.s.get.barrier.state",
    "rocdl.s.get.named.barrier.state",
    "rocdl.s.nop",
    "rocdl.s.setprio",
    "rocdl.s.sleep",
    "rocdl.s.wait.asynccnt",
    "rocdl.s.wait.dscnt",
    "rocdl.s.wait.expcnt",
    "rocdl.s.wait.loadcnt",
    "rocdl.s.wait.storecnt",
    "rocdl.s.wait.tensorcnt",
    "rocdl.s.waitcnt",
    "rocdl.s.wakeup.barrier",
    "rocdl.sched.barrier",
    "rocdl.sched.group.barrier",
    "rocdl.sdot2",
    "rocdl.sdot4",
    "rocdl.sdot8",
    "rocdl.sin",
    "rocdl.smfmac.f32.16x16x128.bf8.bf8",
    "rocdl.smfmac.f32.16x16x128.bf8.fp8",
    "rocdl.smfmac.f32.16x16x128.fp8.bf8",
    "rocdl.smfmac.f32.16x16x128.fp8.fp8",
    "rocdl.smfmac.f32.16x16x32.bf16",
    "rocdl.smfmac.f32.16x16x32.f16",
    "rocdl.smfmac.f32.16x16x64.bf16",
    "rocdl.smfmac.f32.16x16x64.bf8.bf8",
    "rocdl.smfmac.f32.16x16x64.bf8.fp8",
    "rocdl.smfmac.f32.16x16x64.f16",
    "rocdl.smfmac.f32.16x16x64.fp8.bf8",
    "rocdl.smfmac.f32.16x16x64.fp8.fp8",
    "rocdl.smfmac.f32.32x32x16.bf16",
    "rocdl.smfmac.f32.32x32x16.f16",
    "rocdl.smfmac.f32.32x32x32.bf16",
    "rocdl.smfmac.f32.32x32x32.bf8.bf8",
    "rocdl.smfmac.f32.32x32x32.bf8.fp8",
    "rocdl.smfmac.f32.32x32x32.f16",
    "rocdl.smfmac.f32.32x32x32.fp8.bf8",
    "rocdl.smfmac.f32.32x32x32.fp8.fp8",
    "rocdl.smfmac.f32.32x32x64.bf8.bf8",
    "rocdl.smfmac.f32.32x32x64.bf8.fp8",
    "rocdl.smfmac.f32.32x32x64.fp8.bf8",
    "rocdl.smfmac.f32.32x32x64.fp8.fp8",
    "rocdl.smfmac.i32.16x16x128.i8",
    "rocdl.smfmac.i32.16x16x64.i8",
    "rocdl.smfmac.i32.32x32x32.i8",
    "rocdl.smfmac.i32.32x32x64.i8",
    "rocdl.sqrt",
    "rocdl.sudot4",
    "rocdl.sudot8",
    "rocdl.swmmac.bf16.16x16x32.bf16",
    "rocdl.swmmac.bf16.16x16x64.bf16",
    "rocdl.swmmac.bf16f32.16x16x64.bf16",
    "rocdl.swmmac.f16.16x16x128.bf8.bf8",
    "rocdl.swmmac.f16.16x16x128.bf8.fp8",
    "rocdl.swmmac.f16.16x16x128.fp8.bf8",
    "rocdl.swmmac.f16.16x16x128.fp8.fp8",
    "rocdl.swmmac.f16.16x16x32.f16",
    "rocdl.swmmac.f16.16x16x64.f16",
    "rocdl.swmmac.f32.16x16x128.bf8.bf8",
    "rocdl.swmmac.f32.16x16x128.bf8.fp8",
    "rocdl.swmmac.f32.16x16x128.fp8.bf8",
    "rocdl.swmmac.f32.16x16x128.fp8.fp8",
    "rocdl.swmmac.f32.16x16x32.bf16",
    "rocdl.swmmac.f32.16x16x32.bf8.bf8",
    "rocdl.swmmac.f32.16x16x32.bf8.fp8",
    "rocdl.swmmac.f32.16x16x32.f16",
    "rocdl.swmmac.f32.16x16x32.fp8.bf8",
    "rocdl.swmmac.f32.16x16x32.fp8.fp8",
    "rocdl.swmmac.f32.16x16x64.bf16",
    "rocdl.swmmac.f32.16x16x64.f16",
    "rocdl.swmmac.i32.16x16x128.iu8",
    "rocdl.swmmac.i32.16x16x32.iu4",
    "rocdl.swmmac.i32.16x16x32.iu8",
    "rocdl.swmmac.i32.16x16x64.iu4",
    "rocdl.tanh",
    "rocdl.tensor.load.to.lds",
    "rocdl.tensor.store.from.lds",
    "rocdl.udot2",
    "rocdl.udot4",
    "rocdl.udot8",
    "rocdl.update.dpp",
    "rocdl.wait.asyncmark",
    "rocdl.wave.barrier",
    "rocdl.wave.id",
    "rocdl.wavefrontsize",
    "rocdl.wmma.bf16.16x16x16.bf16",
    "rocdl.wmma.bf16.16x16x32.bf16",
    "rocdl.wmma.bf16f32.16x16x32.bf16",
    "rocdl.wmma.f16.16x16x128.bf8_bf8",
    "rocdl.wmma.f16.16x16x128.bf8_fp8",
    "rocdl.wmma.f16.16x16x128.fp8_bf8",
    "rocdl.wmma.f16.16x16x128.fp8_fp8",
    "rocdl.wmma.f16.16x16x16.f16",
    "rocdl.wmma.f16.16x16x32.f16",
    "rocdl.wmma.f16.16x16x64.bf8_bf8",
    "rocdl.wmma.f16.16x16x64.bf8_fp8",
    "rocdl.wmma.f16.16x16x64.fp8_bf8",
    "rocdl.wmma.f16.16x16x64.fp8_fp8",
    "rocdl.wmma.f32.16x16x128.bf8_bf8",
    "rocdl.wmma.f32.16x16x128.bf8_fp8",
    "rocdl.wmma.f32.16x16x128.fp8_bf8",
    "rocdl.wmma.f32.16x16x128.fp8_fp8",
    "rocdl.wmma.f32.16x16x16.bf16",
    "rocdl.wmma.f32.16x16x16.bf8_bf8",
    "rocdl.wmma.f32.16x16x16.bf8_fp8",
    "rocdl.wmma.f32.16x16x16.f16",
    "rocdl.wmma.f32.16x16x16.fp8_bf8",
    "rocdl.wmma.f32.16x16x16.fp8_fp8",
    "rocdl.wmma.f32.16x16x32.bf16",
    "rocdl.wmma.f32.16x16x32.f16",
    "rocdl.wmma.f32.16x16x4.f32",
    "rocdl.wmma.f32.16x16x64.bf8_bf8",
    "rocdl.wmma.f32.16x16x64.bf8_fp8",
    "rocdl.wmma.f32.16x16x64.fp8_bf8",
    "rocdl.wmma.f32.16x16x64.fp8_fp8",
    "rocdl.wmma.i32.16x16x16.iu4",
    "rocdl.wmma.i32.16x16x16.iu8",
    "rocdl.wmma.i32.16x16x32.iu4",
    "rocdl.wmma.i32.16x16x64.iu8",
    "rocdl.wmma.scale.f32.16x16x128.f8f6f4",
    "rocdl.wmma.scale.f32.32x16x128.f4",
    "rocdl.wmma.scale16.f32.16x16x128.f8f6f4",
    "rocdl.wmma.scale16.f32.32x16x128.f4",
    "rocdl.workgroup.id.x",
    "rocdl.workgroup.id.y",
    "rocdl.workgroup.id.z",
    "rocdl.workitem.id.x",
    "rocdl.workitem.id.y",
    "rocdl.workitem.id.z",
};

fn testContext() !*mlir.Context {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    rocdl.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();
    return ctx;
}

fn expectAttributeRoundTrip(ctx: *mlir.Context, attribute: *const mlir.Attribute) !void {
    var buffer: [2048]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buffer);
    try attribute.format(&writer);
    try std.testing.expect(attribute.eql(try mlir.Attribute.parse(ctx, writer.buffered())));
}

test {
    @setEvalBranchQuota(100_000);
    std.testing.refAllDecls(rocdl);
    std.testing.refAllDecls(rocdl.ops);
    std.testing.refAllDecls(rocdl.types);
    std.testing.refAllDecls(rocdl.annotations);
    inline for (rocdl.attributes.enum_attributes) |A| std.testing.refAllDecls(A);
    std.testing.refAllDecls(rocdl.TargetAttr);
}

test "packed FP8 conversion uses the registered LLVM ROCDL dialect" {
    const ctx = try testContext();
    defer ctx.deinit();
    try std.testing.expect(ctx.isRegisteredOperation("rocdl.cvt.pk.f32.fp8"));

    const loc = mlir.Location.unknown(ctx);
    const block = mlir.Block.init(&.{.int(ctx, .i32)}, &.{loc});
    defer block.deinit();
    for ([_]bool{ false, true }) |high| {
        const op = rocdl.cvtPkF32Fp8(ctx, block.argument(0), high, loc);
        defer op.deinit();
        try std.testing.expect(op.verify());
        try std.testing.expect(op.result(0).type_().eql(.vector(&.{2}, .float(ctx, .f32))));
        try std.testing.expect(op.attributeByName("wordSel").?.eql(.boolean(ctx, high)));
    }
}

test "every bound operation is registered" {
    const ctx = try testContext();
    defer ctx.deinit();
    try std.testing.expectEqual(operation_names.len, @as(usize, @intCast(c.mlirROCDLGetOperationCount(ctx.ptr()))));
    for (operation_names, 0..) |name, i| {
        try std.testing.expect(ctx.isRegisteredOperation(name));
        for (operation_names[0..i]) |earlier| try std.testing.expect(!std.mem.eql(u8, name, earlier));
    }
    try std.testing.expect(!ctx.allowUnregisteredDialects());
}

test "all ROCDL enum attributes construct and round-trip every named value" {
    @setEvalBranchQuota(100_000);
    const ctx = try testContext();
    defer ctx.deinit();
    inline for (rocdl.attributes.enum_attributes) |A| {
        for (std.enums.values(A.Value)) |value| {
            const attribute = try A.get(ctx, .{ .value = value });
            try std.testing.expect(attribute.isA(A) != null);
            try std.testing.expectEqual(value, attribute.value());
            try expectAttributeRoundTrip(ctx, attribute.attribute());
        }
    }
    const mask: rocdl.SchedGroupMask = .combine(&.{ .valu, .mfma_wmma, .ds_read });
    try std.testing.expect(mask.contains(.mfma_wmma));
    try std.testing.expect(!mask.contains(.all_vmem));
    const attribute = try rocdl.SchedGroupMaskAttr.get(ctx, .{ .value = mask });
    try std.testing.expectEqual(mask, attribute.value());
    try expectAttributeRoundTrip(ctx, attribute.attribute());
}

test "ROCDL enum constructors reject invalid values and flag bits" {
    const ctx = try testContext();
    defer ctx.deinit();
    try std.testing.expect(c.mlirROCDLMatrixFormatAttrGet(ctx.ptr(), 5).ptr == null);
    try std.testing.expect(c.mlirROCDLWMMACModifierAttrGet(ctx.ptr(), 65536).ptr == null);
    try std.testing.expectError(error.InvalidMlir, rocdl.SchedGroupMaskAttr.get(ctx, .{ .value = @fromBackingInt(1 << 20) }));
    try std.testing.expectError(error.InvalidMlir, rocdl.PreGfx12CachePolicyAttr.get(ctx, .{ .value = @fromBackingInt(1 << 6) }));
}

test "target attributes expose all fields and reject invalid options" {
    const ctx = try testContext();
    defer ctx.deinit();
    const defaults = try rocdl.TargetAttr.get(ctx, .{});
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
    const target = try rocdl.TargetAttr.get(ctx, .{ .optLevel = 3, .chip = "gfx942", .features = "+xnack", .abi = "500", .flags = flags, .link = link });
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

    try std.testing.expectError(error.InvalidMlir, rocdl.TargetAttr.get(ctx, .{ .optLevel = 4 }));
    try std.testing.expectError(error.InvalidMlir, rocdl.TargetAttr.get(ctx, .{ .triple = "" }));
    try std.testing.expectError(error.InvalidMlir, rocdl.TargetAttr.get(ctx, .{ .chip = "" }));
    try std.testing.expectError(error.InvalidMlir, rocdl.TargetAttr.get(ctx, .{ .abi = "700" }));
    try std.testing.expectError(error.InvalidMlir, rocdl.TargetAttr.get(ctx, .{ .flags = .unit(ctx) }));
    const badLink: *const mlir.Attribute = @ptrCast(mlir.ArrayAttribute.init(ctx, &.{.unit(ctx)}));
    try std.testing.expectError(error.InvalidMlir, rocdl.TargetAttr.get(ctx, .{ .link = badLink }));
    const other = try testContext();
    defer other.deinit();
    const otherFlags: *const mlir.Attribute = @ptrCast(mlir.DictionaryAttribute.init(other, &.{}));
    try std.testing.expectError(error.InvalidMlir, rocdl.TargetAttr.get(ctx, .{ .flags = otherFlags }));
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
    const vec4f32 = rocdl.types.v4f32(ctx);
    const zero: *const mlir.Attribute = .int(ctx, .i32, 0);
    const opList = .{
        rocdl.ops.s_barrier(ctx, loc),
        rocdl.ops.s_sleep(ctx, .int(ctx, .i32, 1), loc),
        rocdl.ops.workitem_id_x(ctx, i32Type, .{}, loc),
        rocdl.ops.sched_barrier(ctx, (try rocdl.SchedGroupMaskAttr.get(ctx, .{ .value = .combine(&.{ .valu, .mfma_wmma }) })).attribute(), loc),
        rocdl.ops.raw_buffer_load(ctx, block.argument(0), block.argument(1), block.argument(1), f32Type, .{}, loc),
        rocdl.ops.raw_buffer_load(
            ctx,
            block.argument(0),
            block.argument(1),
            block.argument(1),
            f32Type,
            .{ .aux = (try rocdl.Gfx942CachePolicyAttr.get(ctx, .{ .value = .combine(&.{ .sc0, .nt }) })).attribute() },
            loc,
        ),
        rocdl.ops.mfma_f32_16x16x4f32(ctx, block.argument(2), block.argument(2), block.argument(3), vec4f32, zero, zero, (try rocdl.MFMAPermBAttr.get(ctx, .{ .value = .none })).attribute(), loc),
    };
    inline for (opList) |op| {
        try std.testing.expect(op.verify());
        block.insertOwnedOperationBefore(terminator, op);
    }
    rocdl.annotations.annotateFunction(ctx, function, .{
        .kernel = true,
        .reqdWorkGroupSize = .{ 64, 1, 1 },
        .flatWorkGroupSize = "64,64",
        .maxFlatWorkGroupSize = 64,
        .wavesPerEu = .{ .min = 1, .max = 2 },
        .unsafeFpAtomics = true,
    });
    rocdl.annotations.annotateMemory(ctx, opList[4], .{ .lastUse = true, .noRemoteMemory = true, .noFineGrainedMemory = true, .ignoreDenormalMode = true });
    try std.testing.expect(opList[4].attributeByName("aux").?.eql(.int(ctx, .i32, 0)));
    try std.testing.expect(opList[5].attributeByName("aux").?.isA(rocdl.Gfx942CachePolicyAttr) != null);
    try std.testing.expect(function.attributeByName("rocdl.waves_per_eu").?.eql(.string(ctx, "1,2")));
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
    rocdl.annotations.annotateModule(ctx, module.operation(), .{});
    try std.testing.expect(module.operation().attributeByName("rocdl.xnack") == null);
    rocdl.annotations.annotateModule(ctx, module.operation(), .{ .xnack = false, .sramecc = true });
    try std.testing.expect(module.operation().attributeByName("rocdl.xnack").?.eql(.boolean(ctx, false)));
    try std.testing.expect(module.operation().attributeByName("rocdl.sramecc").?.eql(.boolean(ctx, true)));
    try std.testing.expect(module.operation().verify());
    const pointer = rocdl.pointerType(ctx, .barrier);
    try std.testing.expect(pointer.eql(try mlir.Type.parse(ctx, "!llvm.ptr<15>")));
    try std.testing.expect(rocdl.pointerType(ctx, .buffer_resource).eql(try mlir.Type.parse(ctx, "!llvm.ptr<8>")));
}
