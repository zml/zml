// Generated from LLVM ROCDL TableGen records; see REGENERATING.md.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

const mlir = @import("mlir");
const stdx = @import("stdx");

pub const names = [_][]const u8{
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

/// `rocdl.asyncmark`. Mark the end of a group of asynchronous operations
pub fn asyncmark(
    ctx: *mlir.Context,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.asyncmark", .{
        .location = location,
    });
}

/// `rocdl.ballot`. Vote across thread group
pub fn ballot(
    ctx: *mlir.Context,
    pred: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.ballot", .{
        .operands = .{ .flat = &.{pred} },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.barrier`.
pub fn barrier(
    ctx: *mlir.Context,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.barrier", .{
        .location = location,
    });
}

/// `rocdl.cluster.id.x`.
pub fn cluster_id_x(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "range", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.cluster.id.x", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cluster.id.y`.
pub fn cluster_id_y(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "range", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.cluster.id.y", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cluster.id.z`.
pub fn cluster_id_z(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "range", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.cluster.id.z", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cluster.load.async.to.lds.b128`.
pub fn cluster_load_async_to_lds_b128(
    ctx: *mlir.Context,
    globalPtr: *const mlir.Value,
    ldsPtr: *const mlir.Value,
    mask: *const mlir.Value,
    offset: *const mlir.Attribute,
    opts: struct {
        cpol: ?*const mlir.Attribute = null,
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "offset", offset));
    if (opts.cpol) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "cpol", attribute));
    }
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.cluster.load.async.to.lds.b128", .{
        .operands = .{ .flat = &.{ globalPtr, ldsPtr, mask } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cluster.load.async.to.lds.b32`.
pub fn cluster_load_async_to_lds_b32(
    ctx: *mlir.Context,
    globalPtr: *const mlir.Value,
    ldsPtr: *const mlir.Value,
    mask: *const mlir.Value,
    offset: *const mlir.Attribute,
    opts: struct {
        cpol: ?*const mlir.Attribute = null,
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "offset", offset));
    if (opts.cpol) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "cpol", attribute));
    }
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.cluster.load.async.to.lds.b32", .{
        .operands = .{ .flat = &.{ globalPtr, ldsPtr, mask } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cluster.load.async.to.lds.b64`.
pub fn cluster_load_async_to_lds_b64(
    ctx: *mlir.Context,
    globalPtr: *const mlir.Value,
    ldsPtr: *const mlir.Value,
    mask: *const mlir.Value,
    offset: *const mlir.Attribute,
    opts: struct {
        cpol: ?*const mlir.Attribute = null,
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "offset", offset));
    if (opts.cpol) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "cpol", attribute));
    }
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.cluster.load.async.to.lds.b64", .{
        .operands = .{ .flat = &.{ globalPtr, ldsPtr, mask } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cluster.load.async.to.lds.b8`.
pub fn cluster_load_async_to_lds_b8(
    ctx: *mlir.Context,
    globalPtr: *const mlir.Value,
    ldsPtr: *const mlir.Value,
    mask: *const mlir.Value,
    offset: *const mlir.Attribute,
    opts: struct {
        cpol: ?*const mlir.Attribute = null,
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "offset", offset));
    if (opts.cpol) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "cpol", attribute));
    }
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.cluster.load.async.to.lds.b8", .{
        .operands = .{ .flat = &.{ globalPtr, ldsPtr, mask } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cluster.workgroup.id.x`.
pub fn cluster_workgroup_id_x(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "range", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.cluster.workgroup.id.x", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cluster.workgroup.id.y`.
pub fn cluster_workgroup_id_y(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "range", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.cluster.workgroup.id.y", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cluster.workgroup.id.z`.
pub fn cluster_workgroup_id_z(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "range", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.cluster.workgroup.id.z", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cos`.
pub fn cos(
    ctx: *mlir.Context,
    arg: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cos", .{
        .operands = .{ .flat = &.{arg} },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.f32.bf8`. Convert bf8 to f32
pub fn cvt_f32_bf8(
    ctx: *mlir.Context,
    srcA: *const mlir.Value,
    res_type: *const mlir.Type,
    byteSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "byteSel", byteSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.f32.bf8", .{
        .operands = .{ .flat = &.{srcA} },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.f32.fp8`. Convert fp8 to f32
pub fn cvt_f32_fp8(
    ctx: *mlir.Context,
    srcA: *const mlir.Value,
    res_type: *const mlir.Type,
    byteSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "byteSel", byteSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.f32.fp8", .{
        .operands = .{ .flat = &.{srcA} },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.pk.bf8.f32`. Convert two f32's to bf8
pub fn cvt_pk_bf8_f32(
    ctx: *mlir.Context,
    srcA: *const mlir.Value,
    srcB: *const mlir.Value,
    old: *const mlir.Value,
    res_type: *const mlir.Type,
    wordSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "wordSel", wordSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.pk.bf8.f32", .{
        .operands = .{ .flat = &.{ srcA, srcB, old } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.pk.f32.bf8`. Convert packed bf8 to packed f32
pub fn cvt_pk_f32_bf8(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    res_type: *const mlir.Type,
    wordSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "wordSel", wordSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.pk.f32.bf8", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.pk.f32.fp8`. Convert packed fp8 to packed f32
pub fn cvt_pk_f32_fp8(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    res_type: *const mlir.Type,
    wordSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "wordSel", wordSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.pk.f32.fp8", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.pk.fp8.f32`. Convert two f32's to fp8
pub fn cvt_pk_fp8_f32(
    ctx: *mlir.Context,
    srcA: *const mlir.Value,
    srcB: *const mlir.Value,
    old: *const mlir.Value,
    res_type: *const mlir.Type,
    wordSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "wordSel", wordSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.pk.fp8.f32", .{
        .operands = .{ .flat = &.{ srcA, srcB, old } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.pkrtz`. Convert two f32 input into a vector<2xf16>
pub fn cvt_pkrtz(
    ctx: *mlir.Context,
    srcA: *const mlir.Value,
    srcB: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.pkrtz", .{
        .operands = .{ .flat = &.{ srcA, srcB } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scale.pk16.bf16.bf6`. Scales 16 bf6 and converts them to 16 bf16.
pub fn cvt_scale_pk16_bf16_bf6(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    scaleSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "scaleSel", scaleSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.scale.pk16.bf16.bf6", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scale.pk16.bf16.fp6`. Scales 16 fp6 and converts them to 16 bf16.
pub fn cvt_scale_pk16_bf16_fp6(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    scaleSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "scaleSel", scaleSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.scale.pk16.bf16.fp6", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scale.pk16.f16.bf6`. Scales 16 bf6 and converts them to 16 f16.
pub fn cvt_scale_pk16_f16_bf6(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    scaleSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "scaleSel", scaleSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.scale.pk16.f16.bf6", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scale.pk16.f16.fp6`. Scales 16 fp6 and converts them to 16 f16.
pub fn cvt_scale_pk16_f16_fp6(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    scaleSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "scaleSel", scaleSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.scale.pk16.f16.fp6", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scale.pk16.f32.bf6`. Scales 16 bf6 and converts them to 16 f32.
pub fn cvt_scale_pk16_f32_bf6(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    scaleSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "scaleSel", scaleSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.scale.pk16.f32.bf6", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scale.pk16.f32.fp6`. Scales 16 fp6 and converts them to 16 f32.
pub fn cvt_scale_pk16_f32_fp6(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    scaleSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "scaleSel", scaleSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.scale.pk16.f32.fp6", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scale.pk8.bf16.bf8`. Scales 8 bf8 and converts them to 8 bf16.
pub fn cvt_scale_pk8_bf16_bf8(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    scaleSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "scaleSel", scaleSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.scale.pk8.bf16.bf8", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scale.pk8.bf16.fp4`. Scales 8 fp4 and converts them to 8 bf16.
pub fn cvt_scale_pk8_bf16_fp4(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    scaleSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "scaleSel", scaleSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.scale.pk8.bf16.fp4", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scale.pk8.bf16.fp8`. Scales 8 fp8 and converts them to 8 bf16.
pub fn cvt_scale_pk8_bf16_fp8(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    scaleSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "scaleSel", scaleSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.scale.pk8.bf16.fp8", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scale.pk8.f16.bf8`. Scales 8 bf8 and converts them to 8 f16.
pub fn cvt_scale_pk8_f16_bf8(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    scaleSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "scaleSel", scaleSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.scale.pk8.f16.bf8", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scale.pk8.f16.fp4`. Scales 8 fp4 and converts them to 8 f16.
pub fn cvt_scale_pk8_f16_fp4(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    scaleSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "scaleSel", scaleSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.scale.pk8.f16.fp4", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scale.pk8.f16.fp8`. Scales 8 fp8 and converts them to 8 f16.
pub fn cvt_scale_pk8_f16_fp8(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    scaleSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "scaleSel", scaleSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.scale.pk8.f16.fp8", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scale.pk8.f32.bf8`. Scales 8 bf8 and converts them to 8 f32.
pub fn cvt_scale_pk8_f32_bf8(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    scaleSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "scaleSel", scaleSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.scale.pk8.f32.bf8", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scale.pk8.f32.fp4`. Scales 8 fp4 and converts them to 8 f32.
pub fn cvt_scale_pk8_f32_fp4(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    scaleSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "scaleSel", scaleSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.scale.pk8.f32.fp4", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scale.pk8.f32.fp8`. Scales 8 fp8 and converts them to 8 f32.
pub fn cvt_scale_pk8_f32_fp8(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    scaleSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "scaleSel", scaleSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.scale.pk8.f32.fp8", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.2xpk16.bf6.f32`. Scale and convert two vector<16xf32> to 32 packed bf6
pub fn cvt_scalef32_2xpk16_bf6_f32(
    ctx: *mlir.Context,
    src0: *const mlir.Value,
    src1: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.2xpk16.bf6.f32", .{
        .operands = .{ .flat = &.{ src0, src1, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.2xpk16.fp6.f32`. Scale and convert two vector<16xf32> to 32 packed fp6
pub fn cvt_scalef32_2xpk16_fp6_f32(
    ctx: *mlir.Context,
    src0: *const mlir.Value,
    src1: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.2xpk16.fp6.f32", .{
        .operands = .{ .flat = &.{ src0, src1, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.f16.bf8`. Scaled convert bf8 from packed vector to f16, updating tied result
pub fn cvt_scalef32_f16_bf8(
    ctx: *mlir.Context,
    oldVdst: *const mlir.Value,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    srcSelIndex: *const mlir.Attribute,
    dstLoHiSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "srcSelIndex", srcSelIndex));
    attributes.appendAssumeCapacity(.named(ctx, "dstLoHiSel", dstLoHiSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.f16.bf8", .{
        .operands = .{ .flat = &.{ oldVdst, src, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.f16.fp8`. Scaled convert fp8 from packed vector to f16, updating tied result
pub fn cvt_scalef32_f16_fp8(
    ctx: *mlir.Context,
    oldVdst: *const mlir.Value,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    srcSelIndex: *const mlir.Attribute,
    dstLoHiSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "srcSelIndex", srcSelIndex));
    attributes.appendAssumeCapacity(.named(ctx, "dstLoHiSel", dstLoHiSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.f16.fp8", .{
        .operands = .{ .flat = &.{ oldVdst, src, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.f32.bf8`. Scaled convert bf8 from packed vector to f32
pub fn cvt_scalef32_f32_bf8(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    srcSelIndex: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "srcSelIndex", srcSelIndex));
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.f32.bf8", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.f32.fp8`. Scaled convert fp8 from packed vector to f32
pub fn cvt_scalef32_f32_fp8(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    srcSelIndex: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "srcSelIndex", srcSelIndex));
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.f32.fp8", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk.bf16.bf8`. Scaled convert two bf8to two bf16
pub fn cvt_scalef32_pk_bf16_bf8(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    srcLoHiSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "srcLoHiSel", srcLoHiSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk.bf16.bf8", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk.bf16.fp4`. Scale and convert two packed fp4 to packed bf16
pub fn cvt_scalef32_pk_bf16_fp4(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    srcSelIndex: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "srcSelIndex", srcSelIndex));
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk.bf16.fp4", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk.bf16.fp8`. Scaled convert two fp8to two bf16
pub fn cvt_scalef32_pk_bf16_fp8(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    srcLoHiSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "srcLoHiSel", srcLoHiSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk.bf16.fp8", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk.bf8.bf16`. Scaled convert two bf16to two bf8, updating packed vector
pub fn cvt_scalef32_pk_bf8_bf16(
    ctx: *mlir.Context,
    oldVdst: *const mlir.Value,
    src0: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    dstLoHiSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "dstLoHiSel", dstLoHiSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk.bf8.bf16", .{
        .operands = .{ .flat = &.{ oldVdst, src0, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk.bf8.f16`. Scaled convert two f16to two bf8, updating packed vector
pub fn cvt_scalef32_pk_bf8_f16(
    ctx: *mlir.Context,
    oldVdst: *const mlir.Value,
    src0: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    dstLoHiSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "dstLoHiSel", dstLoHiSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk.bf8.f16", .{
        .operands = .{ .flat = &.{ oldVdst, src0, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk.bf8.f32`. Scaled convert two f32 to two bf8, updating packed vector
pub fn cvt_scalef32_pk_bf8_f32(
    ctx: *mlir.Context,
    oldVdst: *const mlir.Value,
    src0: *const mlir.Value,
    src1: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    dstLoHiSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "dstLoHiSel", dstLoHiSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk.bf8.f32", .{
        .operands = .{ .flat = &.{ oldVdst, src0, src1, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk.f16.bf8`. Scaled convert two bf8to two f16
pub fn cvt_scalef32_pk_f16_bf8(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    srcLoHiSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "srcLoHiSel", srcLoHiSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk.f16.bf8", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk.f16.fp4`. Scale and convert two packed fp4 to packed f16
pub fn cvt_scalef32_pk_f16_fp4(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    srcSelIndex: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "srcSelIndex", srcSelIndex));
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk.f16.fp4", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk.f16.fp8`. Scaled convert two fp8to two f16
pub fn cvt_scalef32_pk_f16_fp8(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    srcLoHiSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "srcLoHiSel", srcLoHiSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk.f16.fp8", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk.f32.bf8`. Scaled convert two bf8to two f32
pub fn cvt_scalef32_pk_f32_bf8(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    srcLoHiSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "srcLoHiSel", srcLoHiSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk.f32.bf8", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk.f32.fp4`. Scale and convert two packed fp4 to packed f32
pub fn cvt_scalef32_pk_f32_fp4(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    srcSelIndex: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "srcSelIndex", srcSelIndex));
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk.f32.fp4", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk.f32.fp8`. Scaled convert two fp8to two f32
pub fn cvt_scalef32_pk_f32_fp8(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    srcLoHiSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "srcLoHiSel", srcLoHiSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk.f32.fp8", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk.fp4.bf16`. Scale and convert two bf16 to packed fp4, updating tied vector
pub fn cvt_scalef32_pk_fp4_bf16(
    ctx: *mlir.Context,
    oldVdst: *const mlir.Value,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    dstSelIndex: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "dstSelIndex", dstSelIndex));
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk.fp4.bf16", .{
        .operands = .{ .flat = &.{ oldVdst, src, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk.fp4.f16`. Scale and convert two f16 to packed fp4, updating tied vector
pub fn cvt_scalef32_pk_fp4_f16(
    ctx: *mlir.Context,
    oldVdst: *const mlir.Value,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    dstSelIndex: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "dstSelIndex", dstSelIndex));
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk.fp4.f16", .{
        .operands = .{ .flat = &.{ oldVdst, src, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk.fp4.f32`. Scale and convert two f32 values to two packed fp4, updating tied vector
pub fn cvt_scalef32_pk_fp4_f32(
    ctx: *mlir.Context,
    oldVdst: *const mlir.Value,
    src0: *const mlir.Value,
    src1: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    dstSelIndex: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "dstSelIndex", dstSelIndex));
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk.fp4.f32", .{
        .operands = .{ .flat = &.{ oldVdst, src0, src1, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk.fp8.bf16`. Scaled convert two bf16to two fp8, updating packed vector
pub fn cvt_scalef32_pk_fp8_bf16(
    ctx: *mlir.Context,
    oldVdst: *const mlir.Value,
    src0: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    dstLoHiSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "dstLoHiSel", dstLoHiSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk.fp8.bf16", .{
        .operands = .{ .flat = &.{ oldVdst, src0, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk.fp8.f16`. Scaled convert two f16to two fp8, updating packed vector
pub fn cvt_scalef32_pk_fp8_f16(
    ctx: *mlir.Context,
    oldVdst: *const mlir.Value,
    src0: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    dstLoHiSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "dstLoHiSel", dstLoHiSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk.fp8.f16", .{
        .operands = .{ .flat = &.{ oldVdst, src0, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk.fp8.f32`. Scaled convert two f32 to two fp8, updating packed vector
pub fn cvt_scalef32_pk_fp8_f32(
    ctx: *mlir.Context,
    oldVdst: *const mlir.Value,
    src0: *const mlir.Value,
    src1: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    dstLoHiSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "dstLoHiSel", dstLoHiSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk.fp8.f32", .{
        .operands = .{ .flat = &.{ oldVdst, src0, src1, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk16.bf6.bf16`. Scale and convert packed bf16 to packed bf6
pub fn cvt_scalef32_pk16_bf6_bf16(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk16.bf6.bf16", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk16.bf6.f16`. Scale and convert packed f16 to packed bf6
pub fn cvt_scalef32_pk16_bf6_f16(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk16.bf6.f16", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk16.bf6.f32`. Scale and convert packed f32 to packed bf6
pub fn cvt_scalef32_pk16_bf6_f32(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk16.bf6.f32", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk16.fp6.bf16`. Scale and convert packed bf16 to packed fp6
pub fn cvt_scalef32_pk16_fp6_bf16(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk16.fp6.bf16", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk16.fp6.f16`. Scale and convert packed f16 to packed fp6
pub fn cvt_scalef32_pk16_fp6_f16(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk16.fp6.f16", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk16.fp6.f32`. Scale and convert packed f32 to packed fp6
pub fn cvt_scalef32_pk16_fp6_f32(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk16.fp6.f32", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk32.bf16.bf6`. Scale and convert packed bf6 to packed bf16
pub fn cvt_scalef32_pk32_bf16_bf6(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk32.bf16.bf6", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk32.bf16.fp6`. Scale and convert packed fp6 to packed bf16
pub fn cvt_scalef32_pk32_bf16_fp6(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk32.bf16.fp6", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk32.bf6.bf16`. Scale and convert packed bf16 to packed bf6
pub fn cvt_scalef32_pk32_bf6_bf16(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk32.bf6.bf16", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk32.bf6.f16`. Scale and convert packed f16 to packed bf6
pub fn cvt_scalef32_pk32_bf6_f16(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk32.bf6.f16", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk32.f16.bf6`. Scale and convert packed bf6 to packed f16
pub fn cvt_scalef32_pk32_f16_bf6(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk32.f16.bf6", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk32.f16.fp6`. Scale and convert packed fp6 to packed f16
pub fn cvt_scalef32_pk32_f16_fp6(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk32.f16.fp6", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk32.f32.bf6`. Scale and convert packed bf6 to packed f32
pub fn cvt_scalef32_pk32_f32_bf6(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk32.f32.bf6", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk32.f32.fp6`. Scale and convert packed fp6 to packed f32
pub fn cvt_scalef32_pk32_f32_fp6(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk32.f32.fp6", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk32.fp6.bf16`. Scale and convert packed bf16 to packed fp6
pub fn cvt_scalef32_pk32_fp6_bf16(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk32.fp6.bf16", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk32.fp6.f16`. Scale and convert packed f16 to packed fp6
pub fn cvt_scalef32_pk32_fp6_f16(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk32.fp6.f16", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk8.bf8.bf16`. Scale and convert packed bf16 to packed bf8
pub fn cvt_scalef32_pk8_bf8_bf16(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk8.bf8.bf16", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk8.bf8.f16`. Scale and convert packed f16 to packed bf8
pub fn cvt_scalef32_pk8_bf8_f16(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk8.bf8.f16", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk8.bf8.f32`. Scale and convert packed f32 to packed bf8
pub fn cvt_scalef32_pk8_bf8_f32(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk8.bf8.f32", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk8.fp4.bf16`. Scale and convert packed bf16 to packed fp4
pub fn cvt_scalef32_pk8_fp4_bf16(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk8.fp4.bf16", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk8.fp4.f16`. Scale and convert packed f16 to packed fp4
pub fn cvt_scalef32_pk8_fp4_f16(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk8.fp4.f16", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk8.fp4.f32`. Scale and convert packed f32 to packed fp4
pub fn cvt_scalef32_pk8_fp4_f32(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk8.fp4.f32", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk8.fp8.bf16`. Scale and convert packed bf16 to packed fp8
pub fn cvt_scalef32_pk8_fp8_bf16(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk8.fp8.bf16", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk8.fp8.f16`. Scale and convert packed f16 to packed fp8
pub fn cvt_scalef32_pk8_fp8_f16(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk8.fp8.f16", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.pk8.fp8.f32`. Scale and convert packed f32 to packed fp8
pub fn cvt_scalef32_pk8_fp8_f32(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.pk8.fp8.f32", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.sr.bf8.bf16`. Scaled convert bf16to bf8 with stochiastic rounding, updating packed vector
pub fn cvt_scalef32_sr_bf8_bf16(
    ctx: *mlir.Context,
    oldVdst: *const mlir.Value,
    src0: *const mlir.Value,
    seed: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    dstSelIndex: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "dstSelIndex", dstSelIndex));
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.sr.bf8.bf16", .{
        .operands = .{ .flat = &.{ oldVdst, src0, seed, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.sr.bf8.f16`. Scaled convert f16to bf8 with stochiastic rounding, updating packed vector
pub fn cvt_scalef32_sr_bf8_f16(
    ctx: *mlir.Context,
    oldVdst: *const mlir.Value,
    src0: *const mlir.Value,
    seed: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    dstSelIndex: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "dstSelIndex", dstSelIndex));
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.sr.bf8.f16", .{
        .operands = .{ .flat = &.{ oldVdst, src0, seed, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.sr.bf8.f32`. Scaled convert f32to bf8 with stochiastic rounding, updating packed vector
pub fn cvt_scalef32_sr_bf8_f32(
    ctx: *mlir.Context,
    oldVdst: *const mlir.Value,
    src0: *const mlir.Value,
    seed: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    dstSelIndex: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "dstSelIndex", dstSelIndex));
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.sr.bf8.f32", .{
        .operands = .{ .flat = &.{ oldVdst, src0, seed, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.sr.fp8.bf16`. Scaled convert bf16to fp8 with stochiastic rounding, updating packed vector
pub fn cvt_scalef32_sr_fp8_bf16(
    ctx: *mlir.Context,
    oldVdst: *const mlir.Value,
    src0: *const mlir.Value,
    seed: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    dstSelIndex: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "dstSelIndex", dstSelIndex));
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.sr.fp8.bf16", .{
        .operands = .{ .flat = &.{ oldVdst, src0, seed, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.sr.fp8.f16`. Scaled convert f16to fp8 with stochiastic rounding, updating packed vector
pub fn cvt_scalef32_sr_fp8_f16(
    ctx: *mlir.Context,
    oldVdst: *const mlir.Value,
    src0: *const mlir.Value,
    seed: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    dstSelIndex: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "dstSelIndex", dstSelIndex));
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.sr.fp8.f16", .{
        .operands = .{ .flat = &.{ oldVdst, src0, seed, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.sr.fp8.f32`. Scaled convert f32to fp8 with stochiastic rounding, updating packed vector
pub fn cvt_scalef32_sr_fp8_f32(
    ctx: *mlir.Context,
    oldVdst: *const mlir.Value,
    src0: *const mlir.Value,
    seed: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    dstSelIndex: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "dstSelIndex", dstSelIndex));
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.sr.fp8.f32", .{
        .operands = .{ .flat = &.{ oldVdst, src0, seed, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.sr.pk.fp4.bf16`. Scale and convert two bf16 to packed fp4 with stochiastic rounding, updating tied vector
pub fn cvt_scalef32_sr_pk_fp4_bf16(
    ctx: *mlir.Context,
    oldVdst: *const mlir.Value,
    src: *const mlir.Value,
    seed: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    dstSelIndex: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "dstSelIndex", dstSelIndex));
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.sr.pk.fp4.bf16", .{
        .operands = .{ .flat = &.{ oldVdst, src, seed, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.sr.pk.fp4.f16`. Scale and convert two f16 to packed fp4 with stochiastic rounding, updating tied vector
pub fn cvt_scalef32_sr_pk_fp4_f16(
    ctx: *mlir.Context,
    oldVdst: *const mlir.Value,
    src: *const mlir.Value,
    seed: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    dstSelIndex: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "dstSelIndex", dstSelIndex));
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.sr.pk.fp4.f16", .{
        .operands = .{ .flat = &.{ oldVdst, src, seed, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.sr.pk.fp4.f32`. Scale and convert two f32 to packed fp4 with stochiastic rounding, updating tied vector
pub fn cvt_scalef32_sr_pk_fp4_f32(
    ctx: *mlir.Context,
    oldVdst: *const mlir.Value,
    src: *const mlir.Value,
    seed: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    dstSelIndex: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "dstSelIndex", dstSelIndex));
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.sr.pk.fp4.f32", .{
        .operands = .{ .flat = &.{ oldVdst, src, seed, scale } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.sr.pk16.bf6.bf16`. Scale and convert packed bf16 to packed bf6 with stochastic rounding
pub fn cvt_scalef32_sr_pk16_bf6_bf16(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    seed: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.sr.pk16.bf6.bf16", .{
        .operands = .{ .flat = &.{ src, seed, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.sr.pk16.bf6.f16`. Scale and convert packed f16 to packed bf6 with stochastic rounding
pub fn cvt_scalef32_sr_pk16_bf6_f16(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    seed: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.sr.pk16.bf6.f16", .{
        .operands = .{ .flat = &.{ src, seed, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.sr.pk16.bf6.f32`. Scale and convert packed f32 to packed bf6 with stochastic rounding
pub fn cvt_scalef32_sr_pk16_bf6_f32(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    seed: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.sr.pk16.bf6.f32", .{
        .operands = .{ .flat = &.{ src, seed, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.sr.pk16.fp6.bf16`. Scale and convert packed bf16 to packed fp6 with stochastic rounding
pub fn cvt_scalef32_sr_pk16_fp6_bf16(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    seed: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.sr.pk16.fp6.bf16", .{
        .operands = .{ .flat = &.{ src, seed, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.sr.pk16.fp6.f16`. Scale and convert packed f16 to packed fp6 with stochastic rounding
pub fn cvt_scalef32_sr_pk16_fp6_f16(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    seed: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.sr.pk16.fp6.f16", .{
        .operands = .{ .flat = &.{ src, seed, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.sr.pk16.fp6.f32`. Scale and convert packed f32 to packed fp6 with stochastic rounding
pub fn cvt_scalef32_sr_pk16_fp6_f32(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    seed: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.sr.pk16.fp6.f32", .{
        .operands = .{ .flat = &.{ src, seed, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.sr.pk32.bf6.bf16`. Scale and convert packed bf16 to packed bf6 with stochiastic rounding
pub fn cvt_scalef32_sr_pk32_bf6_bf16(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    seed: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.sr.pk32.bf6.bf16", .{
        .operands = .{ .flat = &.{ src, seed, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.sr.pk32.bf6.f16`. Scale and convert packed f16 to packed bf6 with stochiastic rounding
pub fn cvt_scalef32_sr_pk32_bf6_f16(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    seed: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.sr.pk32.bf6.f16", .{
        .operands = .{ .flat = &.{ src, seed, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.sr.pk32.bf6.f32`. Scale and convert packed f32 to packed bf6 with stochiastic rounding
pub fn cvt_scalef32_sr_pk32_bf6_f32(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    seed: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.sr.pk32.bf6.f32", .{
        .operands = .{ .flat = &.{ src, seed, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.sr.pk32.fp6.bf16`. Scale and convert packed bf16 to packed fp6 with stochiastic rounding
pub fn cvt_scalef32_sr_pk32_fp6_bf16(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    seed: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.sr.pk32.fp6.bf16", .{
        .operands = .{ .flat = &.{ src, seed, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.sr.pk32.fp6.f16`. Scale and convert packed f16 to packed fp6 with stochiastic rounding
pub fn cvt_scalef32_sr_pk32_fp6_f16(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    seed: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.sr.pk32.fp6.f16", .{
        .operands = .{ .flat = &.{ src, seed, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.sr.pk32.fp6.f32`. Scale and convert packed f32 to packed fp6 with stochiastic rounding
pub fn cvt_scalef32_sr_pk32_fp6_f32(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    seed: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.sr.pk32.fp6.f32", .{
        .operands = .{ .flat = &.{ src, seed, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.sr.pk8.bf8.bf16`. Scale and convert packed bf16 to packed bf8 with stochastic rounding
pub fn cvt_scalef32_sr_pk8_bf8_bf16(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    seed: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.sr.pk8.bf8.bf16", .{
        .operands = .{ .flat = &.{ src, seed, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.sr.pk8.bf8.f16`. Scale and convert packed f16 to packed bf8 with stochastic rounding
pub fn cvt_scalef32_sr_pk8_bf8_f16(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    seed: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.sr.pk8.bf8.f16", .{
        .operands = .{ .flat = &.{ src, seed, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.sr.pk8.bf8.f32`. Scale and convert packed f32 to packed bf8 with stochastic rounding
pub fn cvt_scalef32_sr_pk8_bf8_f32(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    seed: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.sr.pk8.bf8.f32", .{
        .operands = .{ .flat = &.{ src, seed, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.sr.pk8.fp4.bf16`. Scale and convert packed bf16 to packed fp4 with stochastic rounding
pub fn cvt_scalef32_sr_pk8_fp4_bf16(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    seed: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.sr.pk8.fp4.bf16", .{
        .operands = .{ .flat = &.{ src, seed, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.sr.pk8.fp4.f16`. Scale and convert packed f16 to packed fp4 with stochastic rounding
pub fn cvt_scalef32_sr_pk8_fp4_f16(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    seed: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.sr.pk8.fp4.f16", .{
        .operands = .{ .flat = &.{ src, seed, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.sr.pk8.fp4.f32`. Scale and convert packed f32 to packed fp4 with stochastic rounding
pub fn cvt_scalef32_sr_pk8_fp4_f32(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    seed: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.sr.pk8.fp4.f32", .{
        .operands = .{ .flat = &.{ src, seed, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.sr.pk8.fp8.bf16`. Scale and convert packed bf16 to packed fp8 with stochastic rounding
pub fn cvt_scalef32_sr_pk8_fp8_bf16(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    seed: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.sr.pk8.fp8.bf16", .{
        .operands = .{ .flat = &.{ src, seed, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.sr.pk8.fp8.f16`. Scale and convert packed f16 to packed fp8 with stochastic rounding
pub fn cvt_scalef32_sr_pk8_fp8_f16(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    seed: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.sr.pk8.fp8.f16", .{
        .operands = .{ .flat = &.{ src, seed, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.scalef32.sr.pk8.fp8.f32`. Scale and convert packed f32 to packed fp8 with stochastic rounding
pub fn cvt_scalef32_sr_pk8_fp8_f32(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    seed: *const mlir.Value,
    scale: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.cvt.scalef32.sr.pk8.fp8.f32", .{
        .operands = .{ .flat = &.{ src, seed, scale } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.cvt.sr.bf8.f32`. Convert f32 to bf8, stochiastic rounding
pub fn cvt_sr_bf8_f32(
    ctx: *mlir.Context,
    srcA: *const mlir.Value,
    srcB: *const mlir.Value,
    old: *const mlir.Value,
    res_type: *const mlir.Type,
    byteSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "byteSel", byteSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.sr.bf8.f32", .{
        .operands = .{ .flat = &.{ srcA, srcB, old } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.cvt.sr.fp8.f32`. Convert f32 to fp8, stochiastic rounding
pub fn cvt_sr_fp8_f32(
    ctx: *mlir.Context,
    srcA: *const mlir.Value,
    srcB: *const mlir.Value,
    old: *const mlir.Value,
    res_type: *const mlir.Type,
    byteSel: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "byteSel", byteSel));
    return mlir.Operation.make(ctx, "rocdl.cvt.sr.fp8.f32", .{
        .operands = .{ .flat = &.{ srcA, srcB, old } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.dot4.f32.bf8.bf8`.
pub fn dot4_f32_bf8_bf8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.dot4.f32.bf8.bf8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.dot4.f32.bf8.fp8`.
pub fn dot4_f32_bf8_fp8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.dot4.f32.bf8.fp8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.dot4.f32.fp8.bf8`.
pub fn dot4_f32_fp8_bf8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.dot4.f32.fp8.bf8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.dot4.f32.fp8.fp8`.
pub fn dot4_f32_fp8_fp8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.dot4.f32.fp8.fp8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.ds.atomic.async.barrier.arrive.b64`.
pub fn ds_atomic_async_barrier_arrive_b64(
    ctx: *mlir.Context,
    barrierPtr: *const mlir.Value,
    opts: struct {
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.ds.atomic.async.barrier.arrive.b64", .{
        .operands = .{ .flat = &.{barrierPtr} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.ds.atomic.barrier.arrive.rtn.b64`.
pub fn ds_atomic_barrier_arrive_rtn_b64(
    ctx: *mlir.Context,
    barrierPtr: *const mlir.Value,
    val: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.ds.atomic.barrier.arrive.rtn.b64", .{
        .operands = .{ .flat = &.{ barrierPtr, val } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.ds.load.tr16.b128`. Loads and transposes a matrix from ds memory to registers (available in gfx1250+).
pub fn ds_load_tr16_b128(
    ctx: *mlir.Context,
    ptr: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.ds.load.tr16.b128", .{
        .operands = .{ .flat = &.{ptr} },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.ds.load.tr4.b64`. Loads and transposes a matrix from ds memory to registers (available in gfx1250+).
pub fn ds_load_tr4_b64(
    ctx: *mlir.Context,
    ptr: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.ds.load.tr4.b64", .{
        .operands = .{ .flat = &.{ptr} },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.ds.load.tr6.b96`. Loads and transposes a matrix from ds memory to registers (available in gfx1250+).
pub fn ds_load_tr6_b96(
    ctx: *mlir.Context,
    ptr: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.ds.load.tr6.b96", .{
        .operands = .{ .flat = &.{ptr} },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.ds.load.tr8.b64`. Loads and transposes a matrix from ds memory to registers (available in gfx1250+).
pub fn ds_load_tr8_b64(
    ctx: *mlir.Context,
    ptr: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.ds.load.tr8.b64", .{
        .operands = .{ .flat = &.{ptr} },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.ds.read.tr16.b64`.
pub fn ds_read_tr16_b64(
    ctx: *mlir.Context,
    ptr: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.ds.read.tr16.b64", .{
        .operands = .{ .flat = &.{ptr} },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.ds.read.tr4.b64`.
pub fn ds_read_tr4_b64(
    ctx: *mlir.Context,
    ptr: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.ds.read.tr4.b64", .{
        .operands = .{ .flat = &.{ptr} },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.ds.read.tr6.b96`.
pub fn ds_read_tr6_b96(
    ctx: *mlir.Context,
    ptr: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.ds.read.tr6.b96", .{
        .operands = .{ .flat = &.{ptr} },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.ds.read.tr8.b64`.
pub fn ds_read_tr8_b64(
    ctx: *mlir.Context,
    ptr: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.ds.read.tr8.b64", .{
        .operands = .{ .flat = &.{ptr} },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.ds_bpermute`.
pub fn ds_bpermute(
    ctx: *mlir.Context,
    index: *const mlir.Value,
    src: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.ds_bpermute", .{
        .operands = .{ .flat = &.{ index, src } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.ds_swizzle`.
pub fn ds_swizzle(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    offset: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.ds_swizzle", .{
        .operands = .{ .flat = &.{ src, offset } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.exp`.
pub fn exp(
    ctx: *mlir.Context,
    arg: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.exp", .{
        .operands = .{ .flat = &.{arg} },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.exp2`.
pub fn exp2(
    ctx: *mlir.Context,
    arg: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.exp2", .{
        .operands = .{ .flat = &.{arg} },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.fdot2`.
pub fn fdot2(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        clamp: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.clamp) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "clamp", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.fdot2", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.fdot2.bf16.bf16`.
pub fn fdot2_bf16_bf16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.fdot2.bf16.bf16", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.fdot2.f16.f16`.
pub fn fdot2_f16_f16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.fdot2.f16.f16", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.fdot2.f32.bf16`.
pub fn fdot2_f32_bf16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        clamp: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.clamp) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "clamp", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.fdot2.f32.bf16", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.flat.prefetch`.
pub fn flat_prefetch(
    ctx: *mlir.Context,
    ptr: *const mlir.Value,
    opts: struct {
        cachePolicy: ?*const mlir.Attribute = null,
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    if (opts.cachePolicy) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "cachePolicy", attribute));
    }
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.flat.prefetch", .{
        .operands = .{ .flat = &.{ptr} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.fmed3`. Median of three float/half values
pub fn fmed3(
    ctx: *mlir.Context,
    src0: *const mlir.Value,
    src1: *const mlir.Value,
    src2: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.fmed3", .{
        .operands = .{ .flat = &.{ src0, src1, src2 } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.global.load.async.lds`. Version of rocdl.load.async.to.lds specialized to global pointers
pub fn global_load_async_lds(
    ctx: *mlir.Context,
    globalPtr: *const mlir.Value,
    ldsPtr: *const mlir.Value,
    size: *const mlir.Attribute,
    offset: *const mlir.Attribute,
    opts: struct {
        aux: ?*const mlir.Attribute = null,
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 6) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "size", size));
    attributes.appendAssumeCapacity(.named(ctx, "offset", offset));
    if (opts.aux) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "aux", attribute));
    }
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.global.load.async.lds", .{
        .operands = .{ .flat = &.{ globalPtr, ldsPtr } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.global.load.async.to.lds.b128`.
pub fn global_load_async_to_lds_b128(
    ctx: *mlir.Context,
    globalPtr: *const mlir.Value,
    ldsPtr: *const mlir.Value,
    offset: *const mlir.Attribute,
    opts: struct {
        aux: ?*const mlir.Attribute = null,
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "offset", offset));
    if (opts.aux) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "aux", attribute));
    }
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.global.load.async.to.lds.b128", .{
        .operands = .{ .flat = &.{ globalPtr, ldsPtr } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.global.load.async.to.lds.b32`.
pub fn global_load_async_to_lds_b32(
    ctx: *mlir.Context,
    globalPtr: *const mlir.Value,
    ldsPtr: *const mlir.Value,
    offset: *const mlir.Attribute,
    opts: struct {
        aux: ?*const mlir.Attribute = null,
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "offset", offset));
    if (opts.aux) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "aux", attribute));
    }
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.global.load.async.to.lds.b32", .{
        .operands = .{ .flat = &.{ globalPtr, ldsPtr } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.global.load.async.to.lds.b64`.
pub fn global_load_async_to_lds_b64(
    ctx: *mlir.Context,
    globalPtr: *const mlir.Value,
    ldsPtr: *const mlir.Value,
    offset: *const mlir.Attribute,
    opts: struct {
        aux: ?*const mlir.Attribute = null,
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "offset", offset));
    if (opts.aux) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "aux", attribute));
    }
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.global.load.async.to.lds.b64", .{
        .operands = .{ .flat = &.{ globalPtr, ldsPtr } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.global.load.async.to.lds.b8`.
pub fn global_load_async_to_lds_b8(
    ctx: *mlir.Context,
    globalPtr: *const mlir.Value,
    ldsPtr: *const mlir.Value,
    offset: *const mlir.Attribute,
    opts: struct {
        aux: ?*const mlir.Attribute = null,
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "offset", offset));
    if (opts.aux) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "aux", attribute));
    }
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.global.load.async.to.lds.b8", .{
        .operands = .{ .flat = &.{ globalPtr, ldsPtr } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.global.load.lds`.
pub fn global_load_lds(
    ctx: *mlir.Context,
    globalPtr: *const mlir.Value,
    ldsPtr: *const mlir.Value,
    size: *const mlir.Attribute,
    offset: *const mlir.Attribute,
    opts: struct {
        aux: ?*const mlir.Attribute = null,
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 6) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "size", size));
    attributes.appendAssumeCapacity(.named(ctx, "offset", offset));
    if (opts.aux) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "aux", attribute));
    }
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.global.load.lds", .{
        .operands = .{ .flat = &.{ globalPtr, ldsPtr } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.global.load.tr.b128`. Loads and transposes a matrix from global memory to registers (available in gfx1250+).
pub fn global_load_tr_b128(
    ctx: *mlir.Context,
    ptr: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.global.load.tr.b128", .{
        .operands = .{ .flat = &.{ptr} },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.global.load.tr.b64`. Loads and transposes a matrix from global memory to registers (available in gfx1250+).
pub fn global_load_tr_b64(
    ctx: *mlir.Context,
    ptr: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.global.load.tr.b64", .{
        .operands = .{ .flat = &.{ptr} },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.global.load.tr4.b64`. Loads and transposes a matrix from global memory to registers (available in gfx1250+).
pub fn global_load_tr4_b64(
    ctx: *mlir.Context,
    ptr: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.global.load.tr4.b64", .{
        .operands = .{ .flat = &.{ptr} },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.global.load.tr6.b96`. Loads and transposes a matrix from global memory to registers (available in gfx1250+).
pub fn global_load_tr6_b96(
    ctx: *mlir.Context,
    ptr: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.global.load.tr6.b96", .{
        .operands = .{ .flat = &.{ptr} },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.global.prefetch`.
pub fn global_prefetch(
    ctx: *mlir.Context,
    ptr: *const mlir.Value,
    opts: struct {
        cachePolicy: ?*const mlir.Attribute = null,
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    if (opts.cachePolicy) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "cachePolicy", attribute));
    }
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.global.prefetch", .{
        .operands = .{ .flat = &.{ptr} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.global.store.async.from.lds.b128`.
pub fn global_store_async_from_lds_b128(
    ctx: *mlir.Context,
    globalPtr: *const mlir.Value,
    ldsPtr: *const mlir.Value,
    offset: *const mlir.Attribute,
    opts: struct {
        aux: ?*const mlir.Attribute = null,
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "offset", offset));
    if (opts.aux) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "aux", attribute));
    }
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.global.store.async.from.lds.b128", .{
        .operands = .{ .flat = &.{ globalPtr, ldsPtr } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.global.store.async.from.lds.b32`.
pub fn global_store_async_from_lds_b32(
    ctx: *mlir.Context,
    globalPtr: *const mlir.Value,
    ldsPtr: *const mlir.Value,
    offset: *const mlir.Attribute,
    opts: struct {
        aux: ?*const mlir.Attribute = null,
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "offset", offset));
    if (opts.aux) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "aux", attribute));
    }
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.global.store.async.from.lds.b32", .{
        .operands = .{ .flat = &.{ globalPtr, ldsPtr } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.global.store.async.from.lds.b64`.
pub fn global_store_async_from_lds_b64(
    ctx: *mlir.Context,
    globalPtr: *const mlir.Value,
    ldsPtr: *const mlir.Value,
    offset: *const mlir.Attribute,
    opts: struct {
        aux: ?*const mlir.Attribute = null,
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "offset", offset));
    if (opts.aux) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "aux", attribute));
    }
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.global.store.async.from.lds.b64", .{
        .operands = .{ .flat = &.{ globalPtr, ldsPtr } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.global.store.async.from.lds.b8`.
pub fn global_store_async_from_lds_b8(
    ctx: *mlir.Context,
    globalPtr: *const mlir.Value,
    ldsPtr: *const mlir.Value,
    offset: *const mlir.Attribute,
    opts: struct {
        aux: ?*const mlir.Attribute = null,
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "offset", offset));
    if (opts.aux) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "aux", attribute));
    }
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.global.store.async.from.lds.b8", .{
        .operands = .{ .flat = &.{ globalPtr, ldsPtr } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.iglp.opt`.
pub fn iglp_opt(
    ctx: *mlir.Context,
    variant: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "variant", variant));
    return mlir.Operation.make(ctx, "rocdl.iglp.opt", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.load.async.to.lds`. Gathering load to LDS that requires explicit async memory tracking
pub fn load_async_to_lds(
    ctx: *mlir.Context,
    globalPtr: *const mlir.Value,
    ldsPtr: *const mlir.Value,
    size: *const mlir.Attribute,
    offset: *const mlir.Attribute,
    opts: struct {
        aux: ?*const mlir.Attribute = null,
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 6) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "size", size));
    attributes.appendAssumeCapacity(.named(ctx, "offset", offset));
    if (opts.aux) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "aux", attribute));
    }
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.load.async.to.lds", .{
        .operands = .{ .flat = &.{ globalPtr, ldsPtr } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.load.to.lds`.
pub fn load_to_lds(
    ctx: *mlir.Context,
    globalPtr: *const mlir.Value,
    ldsPtr: *const mlir.Value,
    size: *const mlir.Attribute,
    offset: *const mlir.Attribute,
    opts: struct {
        aux: ?*const mlir.Attribute = null,
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 6) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "size", size));
    attributes.appendAssumeCapacity(.named(ctx, "offset", offset));
    if (opts.aux) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "aux", attribute));
    }
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.load.to.lds", .{
        .operands = .{ .flat = &.{ globalPtr, ldsPtr } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.log`.
pub fn log(
    ctx: *mlir.Context,
    arg: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.log", .{
        .operands = .{ .flat = &.{arg} },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.make.buffer.rsrc`.
pub fn make_buffer_rsrc(
    ctx: *mlir.Context,
    base: *const mlir.Value,
    stride: *const mlir.Value,
    numRecords: *const mlir.Value,
    flags: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.make.buffer.rsrc", .{
        .operands = .{ .flat = &.{ base, stride, numRecords, flags } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.mbcnt.hi`.
pub fn mbcnt_hi(
    ctx: *mlir.Context,
    in0: *const mlir.Value,
    in1: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        arg_attrs: ?*const mlir.Attribute = null,
        res_attrs: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.arg_attrs) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "arg_attrs", attribute));
    }
    if (opts.res_attrs) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "res_attrs", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.mbcnt.hi", .{
        .operands = .{ .flat = &.{ in0, in1 } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mbcnt.lo`.
pub fn mbcnt_lo(
    ctx: *mlir.Context,
    in0: *const mlir.Value,
    in1: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        arg_attrs: ?*const mlir.Attribute = null,
        res_attrs: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.arg_attrs) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "arg_attrs", attribute));
    }
    if (opts.res_attrs) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "res_attrs", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.mbcnt.lo", .{
        .operands = .{ .flat = &.{ in0, in1 } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f32.16x16x16bf16.1k`.
pub fn mfma_f32_16x16x16bf16_1k(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f32.16x16x16bf16.1k", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f32.16x16x16f16`.
pub fn mfma_f32_16x16x16f16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f32.16x16x16f16", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f32.16x16x1f32`.
pub fn mfma_f32_16x16x1f32(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f32.16x16x1f32", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f32.16x16x2bf16`.
pub fn mfma_f32_16x16x2bf16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f32.16x16x2bf16", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f32.16x16x32.bf16`.
pub fn mfma_f32_16x16x32_bf16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f32.16x16x32.bf16", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f32.16x16x32.bf8.bf8`.
pub fn mfma_f32_16x16x32_bf8_bf8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f32.16x16x32.bf8.bf8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f32.16x16x32.bf8.fp8`.
pub fn mfma_f32_16x16x32_bf8_fp8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f32.16x16x32.bf8.fp8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f32.16x16x32.f16`.
pub fn mfma_f32_16x16x32_f16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f32.16x16x32.f16", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f32.16x16x32.fp8.bf8`.
pub fn mfma_f32_16x16x32_fp8_bf8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f32.16x16x32.fp8.bf8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f32.16x16x32.fp8.fp8`.
pub fn mfma_f32_16x16x32_fp8_fp8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f32.16x16x32.fp8.fp8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f32.16x16x4bf16.1k`.
pub fn mfma_f32_16x16x4bf16_1k(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f32.16x16x4bf16.1k", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f32.16x16x4f16`.
pub fn mfma_f32_16x16x4f16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f32.16x16x4f16", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f32.16x16x4f32`.
pub fn mfma_f32_16x16x4f32(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f32.16x16x4f32", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f32.16x16x8.xf32`.
pub fn mfma_f32_16x16x8_xf32(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f32.16x16x8.xf32", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f32.16x16x8bf16`.
pub fn mfma_f32_16x16x8bf16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f32.16x16x8bf16", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f32.32x32x16.bf16`.
pub fn mfma_f32_32x32x16_bf16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f32.32x32x16.bf16", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f32.32x32x16.bf8.bf8`.
pub fn mfma_f32_32x32x16_bf8_bf8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f32.32x32x16.bf8.bf8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f32.32x32x16.bf8.fp8`.
pub fn mfma_f32_32x32x16_bf8_fp8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f32.32x32x16.bf8.fp8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f32.32x32x16.f16`.
pub fn mfma_f32_32x32x16_f16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f32.32x32x16.f16", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f32.32x32x16.fp8.bf8`.
pub fn mfma_f32_32x32x16_fp8_bf8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f32.32x32x16.fp8.bf8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f32.32x32x16.fp8.fp8`.
pub fn mfma_f32_32x32x16_fp8_fp8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f32.32x32x16.fp8.fp8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f32.32x32x1f32`.
pub fn mfma_f32_32x32x1f32(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f32.32x32x1f32", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f32.32x32x2bf16`.
pub fn mfma_f32_32x32x2bf16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f32.32x32x2bf16", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f32.32x32x2f32`.
pub fn mfma_f32_32x32x2f32(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f32.32x32x2f32", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f32.32x32x4.xf32`.
pub fn mfma_f32_32x32x4_xf32(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f32.32x32x4.xf32", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f32.32x32x4bf16`.
pub fn mfma_f32_32x32x4bf16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f32.32x32x4bf16", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f32.32x32x4bf16.1k`.
pub fn mfma_f32_32x32x4bf16_1k(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f32.32x32x4bf16.1k", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f32.32x32x4f16`.
pub fn mfma_f32_32x32x4f16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f32.32x32x4f16", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f32.32x32x8bf16.1k`.
pub fn mfma_f32_32x32x8bf16_1k(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f32.32x32x8bf16.1k", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f32.32x32x8f16`.
pub fn mfma_f32_32x32x8f16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f32.32x32x8f16", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f32.4x4x1f32`.
pub fn mfma_f32_4x4x1f32(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f32.4x4x1f32", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f32.4x4x2bf16`.
pub fn mfma_f32_4x4x2bf16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f32.4x4x2bf16", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f32.4x4x4bf16.1k`.
pub fn mfma_f32_4x4x4bf16_1k(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f32.4x4x4bf16.1k", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f32.4x4x4f16`.
pub fn mfma_f32_4x4x4f16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f32.4x4x4f16", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f64.16x16x4f64`.
pub fn mfma_f64_16x16x4f64(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f64.16x16x4f64", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.f64.4x4x4f64`.
pub fn mfma_f64_4x4x4f64(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.f64.4x4x4f64", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.i32.16x16x16i8`.
pub fn mfma_i32_16x16x16i8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.i32.16x16x16i8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.i32.16x16x32.i8`.
pub fn mfma_i32_16x16x32_i8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.i32.16x16x32.i8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.i32.16x16x4i8`.
pub fn mfma_i32_16x16x4i8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.i32.16x16x4i8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.i32.16x16x64.i8`.
pub fn mfma_i32_16x16x64_i8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.i32.16x16x64.i8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.i32.32x32x16.i8`.
pub fn mfma_i32_32x32x16_i8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.i32.32x32x16.i8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.i32.32x32x32.i8`.
pub fn mfma_i32_32x32x32_i8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.i32.32x32x32.i8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.i32.32x32x4i8`.
pub fn mfma_i32_32x32x4i8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.i32.32x32x4i8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.i32.32x32x8i8`.
pub fn mfma_i32_32x32x8i8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.i32.32x32x8i8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.i32.4x4x4i8`.
pub fn mfma_i32_4x4x4i8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    return mlir.Operation.make(ctx, "rocdl.mfma.i32.4x4x4i8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.scale.f32.16x16x128.f8f6f4`.
pub fn mfma_scale_f32_16x16x128_f8f6f4(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    scaleA: *const mlir.Value,
    scaleB: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    opselA: *const mlir.Attribute,
    opselB: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    attributes.appendAssumeCapacity(.named(ctx, "opselA", opselA));
    attributes.appendAssumeCapacity(.named(ctx, "opselB", opselB));
    return mlir.Operation.make(ctx, "rocdl.mfma.scale.f32.16x16x128.f8f6f4", .{
        .operands = .{ .flat = &.{ a, b, c, scaleA, scaleB } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.mfma.scale.f32.32x32x64.f8f6f4`.
pub fn mfma_scale_f32_32x32x64_f8f6f4(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    scaleA: *const mlir.Value,
    scaleB: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    blgp: *const mlir.Attribute,
    opselA: *const mlir.Attribute,
    opselB: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "blgp", blgp));
    attributes.appendAssumeCapacity(.named(ctx, "opselA", opselA));
    attributes.appendAssumeCapacity(.named(ctx, "opselB", opselB));
    return mlir.Operation.make(ctx, "rocdl.mfma.scale.f32.32x32x64.f8f6f4", .{
        .operands = .{ .flat = &.{ a, b, c, scaleA, scaleB } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.permlane16.swap`.
pub fn permlane16_swap(
    ctx: *mlir.Context,
    old: *const mlir.Value,
    src: *const mlir.Value,
    res_type: *const mlir.Type,
    fi: *const mlir.Attribute,
    boundControl: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "fi", fi));
    attributes.appendAssumeCapacity(.named(ctx, "boundControl", boundControl));
    return mlir.Operation.make(ctx, "rocdl.permlane16.swap", .{
        .operands = .{ .flat = &.{ old, src } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.permlane16.var`.
pub fn permlane16_var(
    ctx: *mlir.Context,
    old: *const mlir.Value,
    src0: *const mlir.Value,
    src1: *const mlir.Value,
    res_type: *const mlir.Type,
    fi: *const mlir.Attribute,
    boundControl: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "fi", fi));
    attributes.appendAssumeCapacity(.named(ctx, "boundControl", boundControl));
    return mlir.Operation.make(ctx, "rocdl.permlane16.var", .{
        .operands = .{ .flat = &.{ old, src0, src1 } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.permlane32.swap`.
pub fn permlane32_swap(
    ctx: *mlir.Context,
    old: *const mlir.Value,
    src: *const mlir.Value,
    res_type: *const mlir.Type,
    fi: *const mlir.Attribute,
    boundControl: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "fi", fi));
    attributes.appendAssumeCapacity(.named(ctx, "boundControl", boundControl));
    return mlir.Operation.make(ctx, "rocdl.permlane32.swap", .{
        .operands = .{ .flat = &.{ old, src } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.permlanex16`.
pub fn permlanex16(
    ctx: *mlir.Context,
    old: *const mlir.Value,
    src0: *const mlir.Value,
    src1: *const mlir.Value,
    src2: *const mlir.Value,
    res_type: *const mlir.Type,
    fi: *const mlir.Attribute,
    boundControl: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "fi", fi));
    attributes.appendAssumeCapacity(.named(ctx, "boundControl", boundControl));
    return mlir.Operation.make(ctx, "rocdl.permlanex16", .{
        .operands = .{ .flat = &.{ old, src0, src1, src2 } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.permlanex16.var`.
pub fn permlanex16_var(
    ctx: *mlir.Context,
    old: *const mlir.Value,
    src0: *const mlir.Value,
    src1: *const mlir.Value,
    res_type: *const mlir.Type,
    fi: *const mlir.Attribute,
    boundControl: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "fi", fi));
    attributes.appendAssumeCapacity(.named(ctx, "boundControl", boundControl));
    return mlir.Operation.make(ctx, "rocdl.permlanex16.var", .{
        .operands = .{ .flat = &.{ old, src0, src1 } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.ptr.s.buffer.load`.
pub fn ptr_s_buffer_load(
    ctx: *mlir.Context,
    rsrc: *const mlir.Value,
    offset: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        aux: ?*const mlir.Attribute = null,
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    if (opts.aux) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "aux", attribute));
    }
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.ptr.s.buffer.load", .{
        .operands = .{ .flat = &.{ rsrc, offset } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.raw.buffer.atomic.cmpswap`.
pub fn raw_buffer_atomic_cmpswap(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    cmp: *const mlir.Value,
    rsrc: *const mlir.Value,
    offset: *const mlir.Value,
    soffset: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        aux: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.aux) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "aux", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.raw.buffer.atomic.cmpswap", .{
        .operands = .{ .flat = &.{ src, cmp, rsrc, offset, soffset } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.raw.buffer.atomic.fadd`.
pub fn raw_buffer_atomic_fadd(
    ctx: *mlir.Context,
    vdata: *const mlir.Value,
    rsrc: *const mlir.Value,
    offset: *const mlir.Value,
    soffset: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        aux: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.aux) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "aux", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.raw.buffer.atomic.fadd", .{
        .operands = .{ .flat = &.{ vdata, rsrc, offset, soffset } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.raw.buffer.atomic.fmax`.
pub fn raw_buffer_atomic_fmax(
    ctx: *mlir.Context,
    vdata: *const mlir.Value,
    rsrc: *const mlir.Value,
    offset: *const mlir.Value,
    soffset: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        aux: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.aux) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "aux", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.raw.buffer.atomic.fmax", .{
        .operands = .{ .flat = &.{ vdata, rsrc, offset, soffset } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.raw.buffer.atomic.smax`.
pub fn raw_buffer_atomic_smax(
    ctx: *mlir.Context,
    vdata: *const mlir.Value,
    rsrc: *const mlir.Value,
    offset: *const mlir.Value,
    soffset: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        aux: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.aux) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "aux", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.raw.buffer.atomic.smax", .{
        .operands = .{ .flat = &.{ vdata, rsrc, offset, soffset } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.raw.buffer.atomic.umin`.
pub fn raw_buffer_atomic_umin(
    ctx: *mlir.Context,
    vdata: *const mlir.Value,
    rsrc: *const mlir.Value,
    offset: *const mlir.Value,
    soffset: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        aux: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.aux) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "aux", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.raw.buffer.atomic.umin", .{
        .operands = .{ .flat = &.{ vdata, rsrc, offset, soffset } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.raw.buffer.load`.
pub fn raw_buffer_load(
    ctx: *mlir.Context,
    rsrc: *const mlir.Value,
    offset: *const mlir.Value,
    soffset: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        aux: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.aux) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "aux", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.raw.buffer.load", .{
        .operands = .{ .flat = &.{ rsrc, offset, soffset } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.raw.buffer.store`.
pub fn raw_buffer_store(
    ctx: *mlir.Context,
    vdata: *const mlir.Value,
    rsrc: *const mlir.Value,
    offset: *const mlir.Value,
    soffset: *const mlir.Value,
    opts: struct {
        aux: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.aux) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "aux", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.raw.buffer.store", .{
        .operands = .{ .flat = &.{ vdata, rsrc, offset, soffset } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.raw.ptr.buffer.atomic.cmpswap`.
pub fn raw_ptr_buffer_atomic_cmpswap(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    cmp: *const mlir.Value,
    rsrc: *const mlir.Value,
    offset: *const mlir.Value,
    soffset: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        aux: ?*const mlir.Attribute = null,
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    if (opts.aux) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "aux", attribute));
    }
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.raw.ptr.buffer.atomic.cmpswap", .{
        .operands = .{ .flat = &.{ src, cmp, rsrc, offset, soffset } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.raw.ptr.buffer.atomic.fadd`.
pub fn raw_ptr_buffer_atomic_fadd(
    ctx: *mlir.Context,
    vdata: *const mlir.Value,
    rsrc: *const mlir.Value,
    offset: *const mlir.Value,
    soffset: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        aux: ?*const mlir.Attribute = null,
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    if (opts.aux) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "aux", attribute));
    }
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.raw.ptr.buffer.atomic.fadd", .{
        .operands = .{ .flat = &.{ vdata, rsrc, offset, soffset } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.raw.ptr.buffer.atomic.fmax`.
pub fn raw_ptr_buffer_atomic_fmax(
    ctx: *mlir.Context,
    vdata: *const mlir.Value,
    rsrc: *const mlir.Value,
    offset: *const mlir.Value,
    soffset: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        aux: ?*const mlir.Attribute = null,
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    if (opts.aux) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "aux", attribute));
    }
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.raw.ptr.buffer.atomic.fmax", .{
        .operands = .{ .flat = &.{ vdata, rsrc, offset, soffset } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.raw.ptr.buffer.atomic.smax`.
pub fn raw_ptr_buffer_atomic_smax(
    ctx: *mlir.Context,
    vdata: *const mlir.Value,
    rsrc: *const mlir.Value,
    offset: *const mlir.Value,
    soffset: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        aux: ?*const mlir.Attribute = null,
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    if (opts.aux) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "aux", attribute));
    }
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.raw.ptr.buffer.atomic.smax", .{
        .operands = .{ .flat = &.{ vdata, rsrc, offset, soffset } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.raw.ptr.buffer.atomic.umin`.
pub fn raw_ptr_buffer_atomic_umin(
    ctx: *mlir.Context,
    vdata: *const mlir.Value,
    rsrc: *const mlir.Value,
    offset: *const mlir.Value,
    soffset: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        aux: ?*const mlir.Attribute = null,
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    if (opts.aux) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "aux", attribute));
    }
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.raw.ptr.buffer.atomic.umin", .{
        .operands = .{ .flat = &.{ vdata, rsrc, offset, soffset } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.raw.ptr.buffer.load`.
pub fn raw_ptr_buffer_load(
    ctx: *mlir.Context,
    rsrc: *const mlir.Value,
    offset: *const mlir.Value,
    soffset: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        aux: ?*const mlir.Attribute = null,
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    if (opts.aux) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "aux", attribute));
    }
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.raw.ptr.buffer.load", .{
        .operands = .{ .flat = &.{ rsrc, offset, soffset } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.raw.ptr.buffer.load.async.lds`. Async variant of raw.ptr.buffer.load.lds
pub fn raw_ptr_buffer_load_async_lds(
    ctx: *mlir.Context,
    rsrc: *const mlir.Value,
    ldsPtr: *const mlir.Value,
    size: *const mlir.Value,
    voffset: *const mlir.Value,
    soffset: *const mlir.Value,
    offset: *const mlir.Value,
    opts: struct {
        aux: ?*const mlir.Attribute = null,
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    if (opts.aux) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "aux", attribute));
    }
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.raw.ptr.buffer.load.async.lds", .{
        .operands = .{ .flat = &.{ rsrc, ldsPtr, size, voffset, soffset, offset } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.raw.ptr.buffer.load.lds`.
pub fn raw_ptr_buffer_load_lds(
    ctx: *mlir.Context,
    rsrc: *const mlir.Value,
    ldsPtr: *const mlir.Value,
    size: *const mlir.Value,
    voffset: *const mlir.Value,
    soffset: *const mlir.Value,
    offset: *const mlir.Value,
    opts: struct {
        aux: ?*const mlir.Attribute = null,
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    if (opts.aux) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "aux", attribute));
    }
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.raw.ptr.buffer.load.lds", .{
        .operands = .{ .flat = &.{ rsrc, ldsPtr, size, voffset, soffset, offset } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.raw.ptr.buffer.store`.
pub fn raw_ptr_buffer_store(
    ctx: *mlir.Context,
    vdata: *const mlir.Value,
    rsrc: *const mlir.Value,
    offset: *const mlir.Value,
    soffset: *const mlir.Value,
    opts: struct {
        aux: ?*const mlir.Attribute = null,
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    if (opts.aux) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "aux", attribute));
    }
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.raw.ptr.buffer.store", .{
        .operands = .{ .flat = &.{ vdata, rsrc, offset, soffset } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.rcp`.
pub fn rcp(
    ctx: *mlir.Context,
    arg: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.rcp", .{
        .operands = .{ .flat = &.{arg} },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.readfirstlane`. Get the value in first active lane.
pub fn readfirstlane(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.readfirstlane", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.readlane`. Get the value in the specific lane.
pub fn readlane(
    ctx: *mlir.Context,
    src0: *const mlir.Value,
    src1: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.readlane", .{
        .operands = .{ .flat = &.{ src0, src1 } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.rsq`.
pub fn rsq(
    ctx: *mlir.Context,
    arg: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.rsq", .{
        .operands = .{ .flat = &.{arg} },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.s.barrier`.
pub fn s_barrier(
    ctx: *mlir.Context,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.s.barrier", .{
        .location = location,
    });
}

/// `rocdl.s.barrier.init`.
pub fn s_barrier_init(
    ctx: *mlir.Context,
    ptr: *const mlir.Value,
    memberCnt: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "memberCnt", memberCnt));
    return mlir.Operation.make(ctx, "rocdl.s.barrier.init", .{
        .operands = .{ .flat = &.{ptr} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.s.barrier.join`.
pub fn s_barrier_join(
    ctx: *mlir.Context,
    ptr: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.s.barrier.join", .{
        .operands = .{ .flat = &.{ptr} },
        .location = location,
    });
}

/// `rocdl.s.barrier.leave`.
pub fn s_barrier_leave(
    ctx: *mlir.Context,
    id: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "id", id));
    return mlir.Operation.make(ctx, "rocdl.s.barrier.leave", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.s.barrier.signal`.
pub fn s_barrier_signal(
    ctx: *mlir.Context,
    id: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "id", id));
    return mlir.Operation.make(ctx, "rocdl.s.barrier.signal", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.s.barrier.signal.isfirst`.
pub fn s_barrier_signal_isfirst(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    id: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "id", id));
    return mlir.Operation.make(ctx, "rocdl.s.barrier.signal.isfirst", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.s.barrier.signal.var`.
pub fn s_barrier_signal_var(
    ctx: *mlir.Context,
    ptr: *const mlir.Value,
    memberCnt: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "memberCnt", memberCnt));
    return mlir.Operation.make(ctx, "rocdl.s.barrier.signal.var", .{
        .operands = .{ .flat = &.{ptr} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.s.barrier.wait`.
pub fn s_barrier_wait(
    ctx: *mlir.Context,
    id: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "id", id));
    return mlir.Operation.make(ctx, "rocdl.s.barrier.wait", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.s.get.barrier.state`.
pub fn s_get_barrier_state(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    id: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "id", id));
    return mlir.Operation.make(ctx, "rocdl.s.get.barrier.state", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.s.get.named.barrier.state`.
pub fn s_get_named_barrier_state(
    ctx: *mlir.Context,
    ptr: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.s.get.named.barrier.state", .{
        .operands = .{ .flat = &.{ptr} },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.s.nop`.
pub fn s_nop(
    ctx: *mlir.Context,
    count: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "count", count));
    return mlir.Operation.make(ctx, "rocdl.s.nop", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.s.setprio`.
pub fn s_setprio(
    ctx: *mlir.Context,
    priority: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "priority", priority));
    return mlir.Operation.make(ctx, "rocdl.s.setprio", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.s.sleep`.
pub fn s_sleep(
    ctx: *mlir.Context,
    count: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "count", count));
    return mlir.Operation.make(ctx, "rocdl.s.sleep", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.s.wait.asynccnt`. Wait until ASYNCCNT is less than or equal to `count`
pub fn s_wait_asynccnt(
    ctx: *mlir.Context,
    count: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "count", count));
    return mlir.Operation.make(ctx, "rocdl.s.wait.asynccnt", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.s.wait.dscnt`. Wait until DSCNT is less than or equal to `count`
pub fn s_wait_dscnt(
    ctx: *mlir.Context,
    count: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "count", count));
    return mlir.Operation.make(ctx, "rocdl.s.wait.dscnt", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.s.wait.expcnt`. Wait until EXPCNT is less than or equal to `count`
pub fn s_wait_expcnt(
    ctx: *mlir.Context,
    count: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "count", count));
    return mlir.Operation.make(ctx, "rocdl.s.wait.expcnt", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.s.wait.loadcnt`. Wait until LOADCNT is less than or equal to `count`
pub fn s_wait_loadcnt(
    ctx: *mlir.Context,
    count: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "count", count));
    return mlir.Operation.make(ctx, "rocdl.s.wait.loadcnt", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.s.wait.storecnt`. Wait until STORECNT is less than or equal to `count`
pub fn s_wait_storecnt(
    ctx: *mlir.Context,
    count: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "count", count));
    return mlir.Operation.make(ctx, "rocdl.s.wait.storecnt", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.s.wait.tensorcnt`. Wait until TENSORCNT is less than or equal to `count`
pub fn s_wait_tensorcnt(
    ctx: *mlir.Context,
    count: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "count", count));
    return mlir.Operation.make(ctx, "rocdl.s.wait.tensorcnt", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.s.waitcnt`.
pub fn s_waitcnt(
    ctx: *mlir.Context,
    bitfield: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "bitfield", bitfield));
    return mlir.Operation.make(ctx, "rocdl.s.waitcnt", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.s.wakeup.barrier`.
pub fn s_wakeup_barrier(
    ctx: *mlir.Context,
    ptr: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.s.wakeup.barrier", .{
        .operands = .{ .flat = &.{ptr} },
        .location = location,
    });
}

/// `rocdl.sched.barrier`.
pub fn sched_barrier(
    ctx: *mlir.Context,
    mask: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "mask", mask));
    return mlir.Operation.make(ctx, "rocdl.sched.barrier", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.sched.group.barrier`.
pub fn sched_group_barrier(
    ctx: *mlir.Context,
    mask: *const mlir.Attribute,
    size: *const mlir.Attribute,
    groupId: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "mask", mask));
    attributes.appendAssumeCapacity(.named(ctx, "size", size));
    attributes.appendAssumeCapacity(.named(ctx, "groupId", groupId));
    return mlir.Operation.make(ctx, "rocdl.sched.group.barrier", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.sdot2`.
pub fn sdot2(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        clamp: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.clamp) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "clamp", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.sdot2", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.sdot4`.
pub fn sdot4(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        clamp: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.clamp) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "clamp", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.sdot4", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.sdot8`.
pub fn sdot8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        clamp: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.clamp) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "clamp", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.sdot8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.sin`.
pub fn sin(
    ctx: *mlir.Context,
    arg: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.sin", .{
        .operands = .{ .flat = &.{arg} },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.smfmac.f32.16x16x128.bf8.bf8`.
pub fn smfmac_f32_16x16x128_bf8_bf8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    return mlir.Operation.make(ctx, "rocdl.smfmac.f32.16x16x128.bf8.bf8", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.smfmac.f32.16x16x128.bf8.fp8`.
pub fn smfmac_f32_16x16x128_bf8_fp8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    return mlir.Operation.make(ctx, "rocdl.smfmac.f32.16x16x128.bf8.fp8", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.smfmac.f32.16x16x128.fp8.bf8`.
pub fn smfmac_f32_16x16x128_fp8_bf8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    return mlir.Operation.make(ctx, "rocdl.smfmac.f32.16x16x128.fp8.bf8", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.smfmac.f32.16x16x128.fp8.fp8`.
pub fn smfmac_f32_16x16x128_fp8_fp8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    return mlir.Operation.make(ctx, "rocdl.smfmac.f32.16x16x128.fp8.fp8", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.smfmac.f32.16x16x32.bf16`.
pub fn smfmac_f32_16x16x32_bf16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    return mlir.Operation.make(ctx, "rocdl.smfmac.f32.16x16x32.bf16", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.smfmac.f32.16x16x32.f16`.
pub fn smfmac_f32_16x16x32_f16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    return mlir.Operation.make(ctx, "rocdl.smfmac.f32.16x16x32.f16", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.smfmac.f32.16x16x64.bf16`.
pub fn smfmac_f32_16x16x64_bf16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    return mlir.Operation.make(ctx, "rocdl.smfmac.f32.16x16x64.bf16", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.smfmac.f32.16x16x64.bf8.bf8`.
pub fn smfmac_f32_16x16x64_bf8_bf8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    return mlir.Operation.make(ctx, "rocdl.smfmac.f32.16x16x64.bf8.bf8", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.smfmac.f32.16x16x64.bf8.fp8`.
pub fn smfmac_f32_16x16x64_bf8_fp8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    return mlir.Operation.make(ctx, "rocdl.smfmac.f32.16x16x64.bf8.fp8", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.smfmac.f32.16x16x64.f16`.
pub fn smfmac_f32_16x16x64_f16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    return mlir.Operation.make(ctx, "rocdl.smfmac.f32.16x16x64.f16", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.smfmac.f32.16x16x64.fp8.bf8`.
pub fn smfmac_f32_16x16x64_fp8_bf8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    return mlir.Operation.make(ctx, "rocdl.smfmac.f32.16x16x64.fp8.bf8", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.smfmac.f32.16x16x64.fp8.fp8`.
pub fn smfmac_f32_16x16x64_fp8_fp8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    return mlir.Operation.make(ctx, "rocdl.smfmac.f32.16x16x64.fp8.fp8", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.smfmac.f32.32x32x16.bf16`.
pub fn smfmac_f32_32x32x16_bf16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    return mlir.Operation.make(ctx, "rocdl.smfmac.f32.32x32x16.bf16", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.smfmac.f32.32x32x16.f16`.
pub fn smfmac_f32_32x32x16_f16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    return mlir.Operation.make(ctx, "rocdl.smfmac.f32.32x32x16.f16", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.smfmac.f32.32x32x32.bf16`.
pub fn smfmac_f32_32x32x32_bf16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    return mlir.Operation.make(ctx, "rocdl.smfmac.f32.32x32x32.bf16", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.smfmac.f32.32x32x32.bf8.bf8`.
pub fn smfmac_f32_32x32x32_bf8_bf8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    return mlir.Operation.make(ctx, "rocdl.smfmac.f32.32x32x32.bf8.bf8", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.smfmac.f32.32x32x32.bf8.fp8`.
pub fn smfmac_f32_32x32x32_bf8_fp8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    return mlir.Operation.make(ctx, "rocdl.smfmac.f32.32x32x32.bf8.fp8", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.smfmac.f32.32x32x32.f16`.
pub fn smfmac_f32_32x32x32_f16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    return mlir.Operation.make(ctx, "rocdl.smfmac.f32.32x32x32.f16", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.smfmac.f32.32x32x32.fp8.bf8`.
pub fn smfmac_f32_32x32x32_fp8_bf8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    return mlir.Operation.make(ctx, "rocdl.smfmac.f32.32x32x32.fp8.bf8", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.smfmac.f32.32x32x32.fp8.fp8`.
pub fn smfmac_f32_32x32x32_fp8_fp8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    return mlir.Operation.make(ctx, "rocdl.smfmac.f32.32x32x32.fp8.fp8", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.smfmac.f32.32x32x64.bf8.bf8`.
pub fn smfmac_f32_32x32x64_bf8_bf8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    return mlir.Operation.make(ctx, "rocdl.smfmac.f32.32x32x64.bf8.bf8", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.smfmac.f32.32x32x64.bf8.fp8`.
pub fn smfmac_f32_32x32x64_bf8_fp8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    return mlir.Operation.make(ctx, "rocdl.smfmac.f32.32x32x64.bf8.fp8", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.smfmac.f32.32x32x64.fp8.bf8`.
pub fn smfmac_f32_32x32x64_fp8_bf8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    return mlir.Operation.make(ctx, "rocdl.smfmac.f32.32x32x64.fp8.bf8", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.smfmac.f32.32x32x64.fp8.fp8`.
pub fn smfmac_f32_32x32x64_fp8_fp8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    return mlir.Operation.make(ctx, "rocdl.smfmac.f32.32x32x64.fp8.fp8", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.smfmac.i32.16x16x128.i8`.
pub fn smfmac_i32_16x16x128_i8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    return mlir.Operation.make(ctx, "rocdl.smfmac.i32.16x16x128.i8", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.smfmac.i32.16x16x64.i8`.
pub fn smfmac_i32_16x16x64_i8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    return mlir.Operation.make(ctx, "rocdl.smfmac.i32.16x16x64.i8", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.smfmac.i32.32x32x32.i8`.
pub fn smfmac_i32_32x32x32_i8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    return mlir.Operation.make(ctx, "rocdl.smfmac.i32.32x32x32.i8", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.smfmac.i32.32x32x64.i8`.
pub fn smfmac_i32_32x32x64_i8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    cbsz: *const mlir.Attribute,
    abid: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "cbsz", cbsz));
    attributes.appendAssumeCapacity(.named(ctx, "abid", abid));
    return mlir.Operation.make(ctx, "rocdl.smfmac.i32.32x32x64.i8", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.sqrt`.
pub fn sqrt(
    ctx: *mlir.Context,
    arg: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.sqrt", .{
        .operands = .{ .flat = &.{arg} },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.sudot4`.
pub fn sudot4(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        signA: ?*const mlir.Attribute = null,
        signB: ?*const mlir.Attribute = null,
        clamp: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.signA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "signA", attribute));
    }
    if (opts.signB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "signB", attribute));
    }
    if (opts.clamp) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "clamp", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.sudot4", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.sudot8`.
pub fn sudot8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        signA: ?*const mlir.Attribute = null,
        signB: ?*const mlir.Attribute = null,
        clamp: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.signA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "signA", attribute));
    }
    if (opts.signB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "signB", attribute));
    }
    if (opts.clamp) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "clamp", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.sudot8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.swmmac.bf16.16x16x32.bf16`.
pub fn swmmac_bf16_16x16x32_bf16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.swmmac.bf16.16x16x32.bf16", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.swmmac.bf16.16x16x64.bf16`.
pub fn swmmac_bf16_16x16x64_bf16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        signA: ?*const mlir.Attribute = null,
        signB: ?*const mlir.Attribute = null,
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    if (opts.signA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "signA", attribute));
    }
    if (opts.signB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "signB", attribute));
    }
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.swmmac.bf16.16x16x64.bf16", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.swmmac.bf16f32.16x16x64.bf16`.
pub fn swmmac_bf16f32_16x16x64_bf16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        signA: ?*const mlir.Attribute = null,
        signB: ?*const mlir.Attribute = null,
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    if (opts.signA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "signA", attribute));
    }
    if (opts.signB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "signB", attribute));
    }
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.swmmac.bf16f32.16x16x64.bf16", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.swmmac.f16.16x16x128.bf8.bf8`.
pub fn swmmac_f16_16x16x128_bf8_bf8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.swmmac.f16.16x16x128.bf8.bf8", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.swmmac.f16.16x16x128.bf8.fp8`.
pub fn swmmac_f16_16x16x128_bf8_fp8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.swmmac.f16.16x16x128.bf8.fp8", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.swmmac.f16.16x16x128.fp8.bf8`.
pub fn swmmac_f16_16x16x128_fp8_bf8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.swmmac.f16.16x16x128.fp8.bf8", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.swmmac.f16.16x16x128.fp8.fp8`.
pub fn swmmac_f16_16x16x128_fp8_fp8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.swmmac.f16.16x16x128.fp8.fp8", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.swmmac.f16.16x16x32.f16`.
pub fn swmmac_f16_16x16x32_f16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.swmmac.f16.16x16x32.f16", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.swmmac.f16.16x16x64.f16`.
pub fn swmmac_f16_16x16x64_f16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        signA: ?*const mlir.Attribute = null,
        signB: ?*const mlir.Attribute = null,
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    if (opts.signA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "signA", attribute));
    }
    if (opts.signB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "signB", attribute));
    }
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.swmmac.f16.16x16x64.f16", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.swmmac.f32.16x16x128.bf8.bf8`.
pub fn swmmac_f32_16x16x128_bf8_bf8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.swmmac.f32.16x16x128.bf8.bf8", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.swmmac.f32.16x16x128.bf8.fp8`.
pub fn swmmac_f32_16x16x128_bf8_fp8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.swmmac.f32.16x16x128.bf8.fp8", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.swmmac.f32.16x16x128.fp8.bf8`.
pub fn swmmac_f32_16x16x128_fp8_bf8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.swmmac.f32.16x16x128.fp8.bf8", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.swmmac.f32.16x16x128.fp8.fp8`.
pub fn swmmac_f32_16x16x128_fp8_fp8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.swmmac.f32.16x16x128.fp8.fp8", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.swmmac.f32.16x16x32.bf16`.
pub fn swmmac_f32_16x16x32_bf16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.swmmac.f32.16x16x32.bf16", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.swmmac.f32.16x16x32.bf8.bf8`.
pub fn swmmac_f32_16x16x32_bf8_bf8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.swmmac.f32.16x16x32.bf8.bf8", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.swmmac.f32.16x16x32.bf8.fp8`.
pub fn swmmac_f32_16x16x32_bf8_fp8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.swmmac.f32.16x16x32.bf8.fp8", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.swmmac.f32.16x16x32.f16`.
pub fn swmmac_f32_16x16x32_f16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.swmmac.f32.16x16x32.f16", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.swmmac.f32.16x16x32.fp8.bf8`.
pub fn swmmac_f32_16x16x32_fp8_bf8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.swmmac.f32.16x16x32.fp8.bf8", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.swmmac.f32.16x16x32.fp8.fp8`.
pub fn swmmac_f32_16x16x32_fp8_fp8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.swmmac.f32.16x16x32.fp8.fp8", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.swmmac.f32.16x16x64.bf16`.
pub fn swmmac_f32_16x16x64_bf16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        signA: ?*const mlir.Attribute = null,
        signB: ?*const mlir.Attribute = null,
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    if (opts.signA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "signA", attribute));
    }
    if (opts.signB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "signB", attribute));
    }
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.swmmac.f32.16x16x64.bf16", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.swmmac.f32.16x16x64.f16`.
pub fn swmmac_f32_16x16x64_f16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        signA: ?*const mlir.Attribute = null,
        signB: ?*const mlir.Attribute = null,
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    if (opts.signA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "signA", attribute));
    }
    if (opts.signB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "signB", attribute));
    }
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.swmmac.f32.16x16x64.f16", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.swmmac.i32.16x16x128.iu8`.
pub fn swmmac_i32_16x16x128_iu8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        signA: ?*const mlir.Attribute = null,
        signB: ?*const mlir.Attribute = null,
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
        clamp: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    if (opts.signA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "signA", attribute));
    }
    if (opts.signB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "signB", attribute));
    }
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    if (opts.clamp) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "clamp", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.swmmac.i32.16x16x128.iu8", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.swmmac.i32.16x16x32.iu4`.
pub fn swmmac_i32_16x16x32_iu4(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        signA: ?*const mlir.Attribute = null,
        signB: ?*const mlir.Attribute = null,
        clamp: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.signA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "signA", attribute));
    }
    if (opts.signB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "signB", attribute));
    }
    if (opts.clamp) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "clamp", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.swmmac.i32.16x16x32.iu4", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.swmmac.i32.16x16x32.iu8`.
pub fn swmmac_i32_16x16x32_iu8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        signA: ?*const mlir.Attribute = null,
        signB: ?*const mlir.Attribute = null,
        clamp: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.signA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "signA", attribute));
    }
    if (opts.signB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "signB", attribute));
    }
    if (opts.clamp) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "clamp", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.swmmac.i32.16x16x32.iu8", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.swmmac.i32.16x16x64.iu4`.
pub fn swmmac_i32_16x16x64_iu4(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    index: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        signA: ?*const mlir.Attribute = null,
        signB: ?*const mlir.Attribute = null,
        clamp: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.signA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "signA", attribute));
    }
    if (opts.signB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "signB", attribute));
    }
    if (opts.clamp) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "clamp", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.swmmac.i32.16x16x64.iu4", .{
        .operands = .{ .flat = &.{ a, b, c, index } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.tanh`.
pub fn tanh(
    ctx: *mlir.Context,
    arg: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.tanh", .{
        .operands = .{ .flat = &.{arg} },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.tensor.load.to.lds`. Base class for ROCDL tensor load/store to/from LDS.
pub fn tensor_load_to_lds(
    ctx: *mlir.Context,
    dgroup0: *const mlir.Value,
    dgroup1: *const mlir.Value,
    dgroup2: *const mlir.Value,
    dgroup3: *const mlir.Value,
    dgroup4: *const mlir.Value,
    opts: struct {
        cachePolicy: ?*const mlir.Attribute = null,
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    if (opts.cachePolicy) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "cachePolicy", attribute));
    }
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.tensor.load.to.lds", .{
        .operands = .{ .flat = &.{ dgroup0, dgroup1, dgroup2, dgroup3, dgroup4 } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.tensor.store.from.lds`. Base class for ROCDL tensor load/store to/from LDS.
pub fn tensor_store_from_lds(
    ctx: *mlir.Context,
    dgroup0: *const mlir.Value,
    dgroup1: *const mlir.Value,
    dgroup2: *const mlir.Value,
    dgroup3: *const mlir.Value,
    dgroup4: *const mlir.Value,
    opts: struct {
        cachePolicy: ?*const mlir.Attribute = null,
        alias_scopes: ?*const mlir.Attribute = null,
        noalias_scopes: ?*const mlir.Attribute = null,
        tbaa: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    if (opts.cachePolicy) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "cachePolicy", attribute));
    }
    if (opts.alias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "alias_scopes", attribute));
    }
    if (opts.noalias_scopes) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "noalias_scopes", attribute));
    }
    if (opts.tbaa) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "tbaa", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.tensor.store.from.lds", .{
        .operands = .{ .flat = &.{ dgroup0, dgroup1, dgroup2, dgroup3, dgroup4 } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.udot2`.
pub fn udot2(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        clamp: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.clamp) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "clamp", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.udot2", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.udot4`.
pub fn udot4(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        clamp: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.clamp) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "clamp", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.udot4", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.udot8`.
pub fn udot8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        clamp: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.clamp) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "clamp", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.udot8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.update.dpp`.
pub fn update_dpp(
    ctx: *mlir.Context,
    old: *const mlir.Value,
    src: *const mlir.Value,
    res_type: *const mlir.Type,
    dppCtrl: *const mlir.Attribute,
    rowMask: *const mlir.Attribute,
    bankMask: *const mlir.Attribute,
    boundCtrl: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "dppCtrl", dppCtrl));
    attributes.appendAssumeCapacity(.named(ctx, "rowMask", rowMask));
    attributes.appendAssumeCapacity(.named(ctx, "bankMask", bankMask));
    attributes.appendAssumeCapacity(.named(ctx, "boundCtrl", boundCtrl));
    return mlir.Operation.make(ctx, "rocdl.update.dpp", .{
        .operands = .{ .flat = &.{ old, src } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wait.asyncmark`. Wait until N or fewer async operation groups are unexecuted
pub fn wait_asyncmark(
    ctx: *mlir.Context,
    count: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "count", count));
    return mlir.Operation.make(ctx, "rocdl.wait.asyncmark", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wave.barrier`.
pub fn wave_barrier(
    ctx: *mlir.Context,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.wave.barrier", .{
        .location = location,
    });
}

/// `rocdl.wave.id`.
pub fn wave_id(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "range", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.wave.id", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wavefrontsize`.
pub fn wavefrontsize(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "range", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.wavefrontsize", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wmma.bf16.16x16x16.bf16`.
pub fn wmma_bf16_16x16x16_bf16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        opsel: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.opsel) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "opsel", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.wmma.bf16.16x16x16.bf16", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wmma.bf16.16x16x32.bf16`.
pub fn wmma_bf16_16x16x32_bf16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        modC: ?*const mlir.Attribute = null,
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.modC) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "modC", attribute));
    }
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.wmma.bf16.16x16x32.bf16", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wmma.bf16f32.16x16x32.bf16`.
pub fn wmma_bf16f32_16x16x32_bf16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        modC: ?*const mlir.Attribute = null,
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.modC) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "modC", attribute));
    }
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.wmma.bf16f32.16x16x32.bf16", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wmma.f16.16x16x128.bf8_bf8`.
pub fn wmma_f16_16x16x128_bf8_bf8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        modC: ?*const mlir.Attribute = null,
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.modC) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "modC", attribute));
    }
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.wmma.f16.16x16x128.bf8_bf8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wmma.f16.16x16x128.bf8_fp8`.
pub fn wmma_f16_16x16x128_bf8_fp8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        modC: ?*const mlir.Attribute = null,
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.modC) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "modC", attribute));
    }
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.wmma.f16.16x16x128.bf8_fp8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wmma.f16.16x16x128.fp8_bf8`.
pub fn wmma_f16_16x16x128_fp8_bf8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        modC: ?*const mlir.Attribute = null,
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.modC) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "modC", attribute));
    }
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.wmma.f16.16x16x128.fp8_bf8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wmma.f16.16x16x128.fp8_fp8`.
pub fn wmma_f16_16x16x128_fp8_fp8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        modC: ?*const mlir.Attribute = null,
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.modC) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "modC", attribute));
    }
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.wmma.f16.16x16x128.fp8_fp8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wmma.f16.16x16x16.f16`.
pub fn wmma_f16_16x16x16_f16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        opsel: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.opsel) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "opsel", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.wmma.f16.16x16x16.f16", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wmma.f16.16x16x32.f16`.
pub fn wmma_f16_16x16x32_f16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        modC: ?*const mlir.Attribute = null,
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.modC) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "modC", attribute));
    }
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.wmma.f16.16x16x32.f16", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wmma.f16.16x16x64.bf8_bf8`.
pub fn wmma_f16_16x16x64_bf8_bf8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        modC: ?*const mlir.Attribute = null,
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.modC) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "modC", attribute));
    }
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.wmma.f16.16x16x64.bf8_bf8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wmma.f16.16x16x64.bf8_fp8`.
pub fn wmma_f16_16x16x64_bf8_fp8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        modC: ?*const mlir.Attribute = null,
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.modC) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "modC", attribute));
    }
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.wmma.f16.16x16x64.bf8_fp8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wmma.f16.16x16x64.fp8_bf8`.
pub fn wmma_f16_16x16x64_fp8_bf8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        modC: ?*const mlir.Attribute = null,
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.modC) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "modC", attribute));
    }
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.wmma.f16.16x16x64.fp8_bf8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wmma.f16.16x16x64.fp8_fp8`.
pub fn wmma_f16_16x16x64_fp8_fp8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        modC: ?*const mlir.Attribute = null,
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.modC) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "modC", attribute));
    }
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.wmma.f16.16x16x64.fp8_fp8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wmma.f32.16x16x128.bf8_bf8`.
pub fn wmma_f32_16x16x128_bf8_bf8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        modC: ?*const mlir.Attribute = null,
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.modC) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "modC", attribute));
    }
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.wmma.f32.16x16x128.bf8_bf8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wmma.f32.16x16x128.bf8_fp8`.
pub fn wmma_f32_16x16x128_bf8_fp8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        modC: ?*const mlir.Attribute = null,
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.modC) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "modC", attribute));
    }
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.wmma.f32.16x16x128.bf8_fp8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wmma.f32.16x16x128.fp8_bf8`.
pub fn wmma_f32_16x16x128_fp8_bf8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        modC: ?*const mlir.Attribute = null,
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.modC) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "modC", attribute));
    }
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.wmma.f32.16x16x128.fp8_bf8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wmma.f32.16x16x128.fp8_fp8`.
pub fn wmma_f32_16x16x128_fp8_fp8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        modC: ?*const mlir.Attribute = null,
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.modC) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "modC", attribute));
    }
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.wmma.f32.16x16x128.fp8_fp8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wmma.f32.16x16x16.bf16`.
pub fn wmma_f32_16x16x16_bf16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.wmma.f32.16x16x16.bf16", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.wmma.f32.16x16x16.bf8_bf8`.
pub fn wmma_f32_16x16x16_bf8_bf8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.wmma.f32.16x16x16.bf8_bf8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.wmma.f32.16x16x16.bf8_fp8`.
pub fn wmma_f32_16x16x16_bf8_fp8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.wmma.f32.16x16x16.bf8_fp8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.wmma.f32.16x16x16.f16`.
pub fn wmma_f32_16x16x16_f16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.wmma.f32.16x16x16.f16", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.wmma.f32.16x16x16.fp8_bf8`.
pub fn wmma_f32_16x16x16_fp8_bf8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.wmma.f32.16x16x16.fp8_bf8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.wmma.f32.16x16x16.fp8_fp8`.
pub fn wmma_f32_16x16x16_fp8_fp8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "rocdl.wmma.f32.16x16x16.fp8_fp8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `rocdl.wmma.f32.16x16x32.bf16`.
pub fn wmma_f32_16x16x32_bf16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        modC: ?*const mlir.Attribute = null,
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.modC) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "modC", attribute));
    }
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.wmma.f32.16x16x32.bf16", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wmma.f32.16x16x32.f16`.
pub fn wmma_f32_16x16x32_f16(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        modC: ?*const mlir.Attribute = null,
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.modC) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "modC", attribute));
    }
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.wmma.f32.16x16x32.f16", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wmma.f32.16x16x4.f32`.
pub fn wmma_f32_16x16x4_f32(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        modC: ?*const mlir.Attribute = null,
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.modC) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "modC", attribute));
    }
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.wmma.f32.16x16x4.f32", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wmma.f32.16x16x64.bf8_bf8`.
pub fn wmma_f32_16x16x64_bf8_bf8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        modC: ?*const mlir.Attribute = null,
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.modC) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "modC", attribute));
    }
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.wmma.f32.16x16x64.bf8_bf8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wmma.f32.16x16x64.bf8_fp8`.
pub fn wmma_f32_16x16x64_bf8_fp8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        modC: ?*const mlir.Attribute = null,
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.modC) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "modC", attribute));
    }
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.wmma.f32.16x16x64.bf8_fp8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wmma.f32.16x16x64.fp8_bf8`.
pub fn wmma_f32_16x16x64_fp8_bf8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        modC: ?*const mlir.Attribute = null,
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.modC) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "modC", attribute));
    }
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.wmma.f32.16x16x64.fp8_bf8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wmma.f32.16x16x64.fp8_fp8`.
pub fn wmma_f32_16x16x64_fp8_fp8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        modC: ?*const mlir.Attribute = null,
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.modC) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "modC", attribute));
    }
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.wmma.f32.16x16x64.fp8_fp8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wmma.i32.16x16x16.iu4`.
pub fn wmma_i32_16x16x16_iu4(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        signA: ?*const mlir.Attribute = null,
        signB: ?*const mlir.Attribute = null,
        clamp: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.signA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "signA", attribute));
    }
    if (opts.signB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "signB", attribute));
    }
    if (opts.clamp) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "clamp", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.wmma.i32.16x16x16.iu4", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wmma.i32.16x16x16.iu8`.
pub fn wmma_i32_16x16x16_iu8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        signA: ?*const mlir.Attribute = null,
        signB: ?*const mlir.Attribute = null,
        clamp: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.signA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "signA", attribute));
    }
    if (opts.signB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "signB", attribute));
    }
    if (opts.clamp) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "clamp", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.wmma.i32.16x16x16.iu8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wmma.i32.16x16x32.iu4`.
pub fn wmma_i32_16x16x32_iu4(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        signA: ?*const mlir.Attribute = null,
        signB: ?*const mlir.Attribute = null,
        clamp: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.signA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "signA", attribute));
    }
    if (opts.signB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "signB", attribute));
    }
    if (opts.clamp) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "clamp", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.wmma.i32.16x16x32.iu4", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wmma.i32.16x16x64.iu8`.
pub fn wmma_i32_16x16x64_iu8(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        signA: ?*const mlir.Attribute = null,
        signB: ?*const mlir.Attribute = null,
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
        clamp: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    if (opts.signA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "signA", attribute));
    }
    if (opts.signB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "signB", attribute));
    }
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    if (opts.clamp) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "clamp", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.wmma.i32.16x16x64.iu8", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wmma.scale.f32.16x16x128.f8f6f4`.
pub fn wmma_scale_f32_16x16x128_f8f6f4(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    scaleA: *const mlir.Value,
    scaleB: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        fmtA: ?*const mlir.Attribute = null,
        fmtB: ?*const mlir.Attribute = null,
        modC: ?*const mlir.Attribute = null,
        scaleAType: ?*const mlir.Attribute = null,
        fmtScaleA: ?*const mlir.Attribute = null,
        scaleBType: ?*const mlir.Attribute = null,
        fmtScaleB: ?*const mlir.Attribute = null,
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 9) = .empty;
    if (opts.fmtA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "fmtA", attribute));
    }
    if (opts.fmtB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "fmtB", attribute));
    }
    if (opts.modC) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "modC", attribute));
    }
    if (opts.scaleAType) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "scaleAType", attribute));
    }
    if (opts.fmtScaleA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "fmtScaleA", attribute));
    }
    if (opts.scaleBType) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "scaleBType", attribute));
    }
    if (opts.fmtScaleB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "fmtScaleB", attribute));
    }
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.wmma.scale.f32.16x16x128.f8f6f4", .{
        .operands = .{ .flat = &.{ a, b, c, scaleA, scaleB } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wmma.scale.f32.32x16x128.f4`.
pub fn wmma_scale_f32_32x16x128_f4(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    scaleA: *const mlir.Value,
    scaleB: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        modC: ?*const mlir.Attribute = null,
        scaleAType: ?*const mlir.Attribute = null,
        fmtScaleA: ?*const mlir.Attribute = null,
        scaleBType: ?*const mlir.Attribute = null,
        fmtScaleB: ?*const mlir.Attribute = null,
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 7) = .empty;
    if (opts.modC) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "modC", attribute));
    }
    if (opts.scaleAType) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "scaleAType", attribute));
    }
    if (opts.fmtScaleA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "fmtScaleA", attribute));
    }
    if (opts.scaleBType) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "scaleBType", attribute));
    }
    if (opts.fmtScaleB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "fmtScaleB", attribute));
    }
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.wmma.scale.f32.32x16x128.f4", .{
        .operands = .{ .flat = &.{ a, b, c, scaleA, scaleB } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wmma.scale16.f32.16x16x128.f8f6f4`.
pub fn wmma_scale16_f32_16x16x128_f8f6f4(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    scaleA: *const mlir.Value,
    scaleB: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        fmtA: ?*const mlir.Attribute = null,
        fmtB: ?*const mlir.Attribute = null,
        modC: ?*const mlir.Attribute = null,
        scaleAType: ?*const mlir.Attribute = null,
        fmtScaleA: ?*const mlir.Attribute = null,
        scaleBType: ?*const mlir.Attribute = null,
        fmtScaleB: ?*const mlir.Attribute = null,
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 9) = .empty;
    if (opts.fmtA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "fmtA", attribute));
    }
    if (opts.fmtB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "fmtB", attribute));
    }
    if (opts.modC) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "modC", attribute));
    }
    if (opts.scaleAType) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "scaleAType", attribute));
    }
    if (opts.fmtScaleA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "fmtScaleA", attribute));
    }
    if (opts.scaleBType) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "scaleBType", attribute));
    }
    if (opts.fmtScaleB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "fmtScaleB", attribute));
    }
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.wmma.scale16.f32.16x16x128.f8f6f4", .{
        .operands = .{ .flat = &.{ a, b, c, scaleA, scaleB } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.wmma.scale16.f32.32x16x128.f4`.
pub fn wmma_scale16_f32_32x16x128_f4(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    scaleA: *const mlir.Value,
    scaleB: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        modC: ?*const mlir.Attribute = null,
        scaleAType: ?*const mlir.Attribute = null,
        fmtScaleA: ?*const mlir.Attribute = null,
        scaleBType: ?*const mlir.Attribute = null,
        fmtScaleB: ?*const mlir.Attribute = null,
        reuseA: ?*const mlir.Attribute = null,
        reuseB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 7) = .empty;
    if (opts.modC) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "modC", attribute));
    }
    if (opts.scaleAType) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "scaleAType", attribute));
    }
    if (opts.fmtScaleA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "fmtScaleA", attribute));
    }
    if (opts.scaleBType) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "scaleBType", attribute));
    }
    if (opts.fmtScaleB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "fmtScaleB", attribute));
    }
    if (opts.reuseA) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseA", attribute));
    }
    if (opts.reuseB) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "reuseB", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.wmma.scale16.f32.32x16x128.f4", .{
        .operands = .{ .flat = &.{ a, b, c, scaleA, scaleB } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.workgroup.id.x`.
pub fn workgroup_id_x(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "range", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.workgroup.id.x", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.workgroup.id.y`.
pub fn workgroup_id_y(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "range", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.workgroup.id.y", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.workgroup.id.z`.
pub fn workgroup_id_z(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "range", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.workgroup.id.z", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.workitem.id.x`.
pub fn workitem_id_x(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "range", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.workitem.id.x", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.workitem.id.y`.
pub fn workitem_id_y(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "range", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.workitem.id.y", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `rocdl.workitem.id.z`.
pub fn workitem_id_z(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |attribute| {
        attributes.appendAssumeCapacity(.named(ctx, "range", attribute));
    }
    return mlir.Operation.make(ctx, "rocdl.workitem.id.z", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}
