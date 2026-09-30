const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");
const stdx = @import("stdx");

const cute = @import("cute_ir.zig");

pub const dialect_namespace = "cute_nvgpu";

pub const NVGPUType = opaque {
    const M = mlir.Methods(NVGPUType, c.MlirType);

    pub const isAFn = c.mlirTypeIsACuteNVGPUType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;

    pub fn format(self: *const NVGPUType, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        const t: *const mlir.Type = @ptrCast(self);
        return t.format(writer);
    }
};

/// Parses `!cute_nvgpu.<T.mnemonic><payload>`, such as `<f32>` for
/// `CopyAtomSIMTSyncCopyType`, and returns it as a `T`.
pub fn typeFromPayload(comptime T: type, ctx: *mlir.Context, payload: []const u8) mlir.Error!*const T {
    var buffer: [1024]u8 = undefined;
    const text = std.fmt.bufPrint(&buffer, "!" ++ dialect_namespace ++ "." ++ T.mnemonic ++ "{s}", .{payload}) catch return error.InvalidMlir;
    return (try mlir.Type.parse(ctx, text)).isA(T) orelse error.InvalidMlir;
}

// Enums, generated from the dialect's .td files; values are the compiler's.

/// NVGpu arch number like sm_103.
pub const Arch = enum(u32) {
    sm_52 = 520,
    sm_70 = 700,
    sm_75 = 750,
    sm_80 = 800,
    sm_86 = 860,
    sm_90 = 900,
    sm_100 = 1000,
    sm_103 = 1030,
    sm_107 = 1070,
};

/// Fields stored in Bulk Load Copy Atom type.
pub const AtomCopyFieldBulkCopyG2S = enum(u32) {
    tma_bar = 1,
    mcast_mask = 2,
    cache_policy = 7,
};

/// Fields stored in Bulk Store Copy Atom type.
pub const AtomCopyFieldBulkCopyS2G = enum(u32) {
    byte_mask = 4,
    cache_policy = 7,
};

/// Fields stored in Bulk CTA to Cluster Copy Atom type.
pub const AtomCopyFieldBulkCopyS2S = enum(u32) {
    tma_bar = 1,
    cta_rank = 5,
};

/// Fields stored in CopyAtomDsmemStoreType.
pub const AtomCopyFieldDsmemStore = enum(u32) {
    mbar_ptr = 8,
};

/// Fields stored in CopyAtomG2RType.
pub const AtomCopyFieldLoadGlobal = enum(u32) {
    cache_policy = 7,
};

/// Fields stored in CopyAtomNonExec2DGather4TmaLoadType.
pub const AtomCopyFieldNonExec2DGather4TmaLoad = enum(u32) {
    tma_desc = 3,
};

/// Fields stored in CopyAtomNonExec2DScatter4TmaStoreType.
pub const AtomCopyFieldNonExec2DScatter4TmaStore = enum(u32) {
    tma_desc = 3,
};

/// Fields stored in CopyAtomNonExecIm2ColTmaLoadType.
pub const AtomCopyFieldNonExecIm2ColTmaLoad = enum(u32) {
    im2col_tma_desc = 6,
};

/// Fields stored in CopyAtomNonExecIm2ColTmaStoreType.
pub const AtomCopyFieldNonExecIm2ColTmaStore = enum(u32) {
    im2col_tma_desc = 6,
};

/// Fields stored in CopyAtomNonExecTiledTmaLoadType.
pub const AtomCopyFieldNonExecTiledTmaLoad = enum(u32) {
    tma_desc = 3,
};

/// Fields stored in CopyAtomNonExecTiledTmaReduceType.
pub const AtomCopyFieldNonExecTiledTmaReduce = enum(u32) {
    tma_desc = 3,
};

/// Fields stored in CopyAtomNonExecTiledTmaStoreType.
pub const AtomCopyFieldNonExecTiledTmaStore = enum(u32) {
    tma_desc = 3,
};

/// Fields stored in CopyAtomR2GType.
pub const AtomCopyFieldStoreGlobal = enum(u32) {
    cache_policy = 7,
};

/// Fields stored in Tma Load Copy Atom type.
pub const AtomCopyFieldTmaLoad = enum(u32) {
    tma_descriptor_ptr = 0,
    tma_bar = 1,
    mcast_mask = 2,
    cache_policy = 7,
    g_stride = 9,
};

/// Fields stored in Tma Reduce Copy Atom type.
pub const AtomCopyFieldTmaReduce = enum(u32) {
    tma_descriptor_ptr = 0,
    cache_policy = 7,
    g_stride = 9,
};

/// Fields stored in Tma Store Copy Atom type.
pub const AtomCopyFieldTmaStore = enum(u32) {
    tma_descriptor_ptr = 0,
    cache_policy = 7,
    g_stride = 9,
};

/// Fields stored in MMA Atom types for SM100.
pub const AtomMmaFieldSM100 = enum(u32) {
    accum_c = 0,
    neg_a = 1,
    neg_b = 2,
    disable_output_lane = 6,
};

/// Fields stored in MMA Atom types for SM100 (block scaled).
pub const AtomMmaFieldSM100BlockScaled = enum(u32) {
    accum_c = 0,
    neg_a = 1,
    neg_b = 2,
    sf_a = 3,
    sf_b = 4,
};

/// Fields stored in MMA Atom types for SM100 (block scaled).
pub const AtomMmaFieldSM100BlockScaledSparse = enum(u32) {
    accum_c = 0,
    neg_a = 1,
    neg_b = 2,
    sf_a = 3,
    sf_b = 4,
    e = 5,
};

/// Fields stored in sparse MMA Atom types for SM100.
pub const AtomMmaFieldSM100Sparse = enum(u32) {
    accum_c = 0,
    neg_a = 1,
    neg_b = 2,
    e = 5,
};

/// Fields stored in MMA Atom types for SM120 (block scaled).
pub const AtomMmaFieldSM120BlockScaled = enum(u32) {
    sf_a = 0,
    sf_b = 1,
};

/// Fields stored in MMA Atom types for SM80.
pub const AtomMmaFieldSM80Sparse = enum(u32) {
    e = 0,
};

/// Fields stored in MMA Atom types for SM90.
pub const AtomMmaFieldSM90 = enum(u32) {
    accum_c = 0,
};

/// Binary operation for single-bit MMA operations.
pub const BinaryOp = enum(u32) {
    none = 0,
    xor_popc = 1,
    and_popc = 2,
};

/// Cute Cache Eviction Priority kind.
pub const CacheEvictionPriority = enum(u32) {
    EVICT_NORMAL = 0,
    EVICT_FIRST = 1,
    EVICT_LAST = 2,
    EVICT_UNCHANGED = 3,
    NO_ALLOCATE = 4,
};

/// Broadcast modes for the different utccp instructions.
pub const CopyS2TBroadcast = enum(u32) {
    none = 0,
    lw_0213 = 1,
    lw_0123 = 2,
    x4 = 3,
};

/// The various kinds of TMA loads in gather/scatter mode.
pub const GatherScatterTmaLoad = enum(u32) {
    sm_100 = 0,
    sm_100_multicast = 1,
    sm_100_2sm = 2,
    sm_100_2sm_multicast = 3,
};

/// The various kinds of TMA loads in im2col mode.
pub const Im2ColTmaLoad = enum(u32) {
    sm_90 = 0,
    sm_90_multicast = 1,
    sm_100_2sm = 2,
    sm_100_2sm_multicast = 3,
};

/// CuTe L2 prefetch size options.
pub const L2PrefetchSize = enum(u32) {
    NONE = 0,
    RESERVED = 1,
    SIZE_64B = 2,
    SIZE_128B = 3,
    SIZE_256B = 4,
};

/// multimem ld_reduce accumulation precision kind.
pub const LdReduceAccPrecisionKind = enum(u32) {
    F16 = 0,
    F32 = 1,
};

/// LDSM's sz pattern, describing the bit size.
pub const LdsmSzPattern = enum(u32) {
    u16 = 0,
    u4to8 = 1,
    s4to8 = 2,
    u2to4 = 3,
    s2to4 = 4,
    u4x16p64to8 = 5,
    u6x16p32to8 = 6,
    u8 = 7,
};

/// Cache modes for the load instructions.
pub const LoadCacheMode = enum(u32) {
    always = 0,
    global = 1,
    streaming = 2,
    last_use = 3,
    none = 4,
};

/// MMA overflow options.
pub const MMAIntOverflow = enum(u32) {
    satfinite = 1,
    wrapped = 0,
};

/// Memory Ordering kind.
pub const MemOrderKind = enum(u32) {
    WEAK = 0,
    RELAXED = 1,
    ACQUIRE = 2,
    RELEASE = 3,
    ACQ_REL = 4,
    SC = 5,
    MMIO = 6,
    CONSTANT = 7,
    VOLATILE = 8,
};

/// Cute Memory Scope kind.
pub const MemScopeKind = enum(u32) {
    CTA = 0,
    CLUSTER = 1,
    GPU = 2,
    SYS = 3,
};

/// Enums for the mma collector op.
pub const MmaCollectorOp = enum(u32) {
    discard = 0,
    lastuse = 1,
    fill = 2,
    use = 3,
};

/// Enums for the mma frag type.
pub const MmaFragKind = enum(u32) {
    smem_desc = 0,
    sparse_smem_desc = 1,
    tmem = 2,
    tmem_ws = 3,
    tmem_e = 4,
    tmem_sf = 5,
    rmem = 6,
};

/// Op for the TMASTORE instruction.
pub const ReductionKind = enum(u32) {
    ADD = 0,
    MIN = 1,
    MAX = 2,
    INC = 3,
    DEC = 4,
    AND = 5,
    OR = 6,
    XOR = 7,
};

/// CuTe Shared Space kind.
pub const SharedSpace = enum(u32) {
    CTA = 0,
    CLUSTER = 1,
};

/// Metadata format used in sparse MMA.
pub const SparseMetadataFormat = enum(u32) {
    tid = 0,
};

/// Cache modes for the store instructions.
pub const StoreCacheMode = enum(u32) {
    write_back = 0,
    global = 1,
    streaming = 2,
    write_through = 3,
    none = 4,
};

/// The various kinds of TMA loads in tiled mode.
pub const TiledTmaLoad = enum(u32) {
    sm_90 = 0,
    sm_90_multicast = 1,
    sm_100_2sm = 2,
    sm_100_2sm_multicast = 3,
};

/// Bits 74-71 of the TMA descriptor.
pub const TmaDataFormat = enum(u32) {
    U8 = 0,
    U16 = 1,
    U32 = 2,
    S32 = 3,
    U64 = 4,
    S64 = 5,
    F16_RN = 6,
    F32_RN = 7,
    F32_FTZ_RN = 8,
    F64_RN = 9,
    BF16_RN = 10,
    U4 = 11,
    U4_UNPACK_U8 = 12,
    U6_UNPACK_U8 = 13,
    TF32_RN = 14,
    TF32_FTZ_RN = 15,
};

/// Modes for the TMA load instruction.
pub const TmaLoadMode = enum(u32) {
    tiled = 0,
    im2col = 1,
    w128 = 2,
    w = 3,
    gather4 = 4,
};

/// Modes for the TMA store instruction.
pub const TmaStoreMode = enum(u32) {
    tiled = 0,
    im2col = 1,
    scatter4 = 2,
};

/// Modes for the Tmem allocation.
pub const TmemAllocMode = enum(u32) {
    Interleaved = 0,
    NonInterleaved = 1,
    Duplicated = 2,
    SfDuplicated_4x1 = 3,
    SfDuplicated_2x2 = 4,
};

/// Reduce operation for the Tmem load instruction.
pub const TmemLoadRedOp = enum(u32) {
    max = 0,
    maxabs = 1,
    min = 2,
    minabs = 3,
};

// Types and attributes, generated from the dialect's .td files.

/// `!cute_nvgpu.atom.bulk_copy_g2s`: Bulk (TMA non-tensor) copy atom.
pub const CopyAtomBulkCopyG2SType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomBulkCopyG2S;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.bulk_copy_g2s";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int = 0,
        mcast: bool = false,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomBulkCopyG2STypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits, args.mcast);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomBulkCopyG2STypeGetValType(self.ptr()).ptr.?);
    }
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomBulkCopyG2STypeGetCopyBits(self.ptr());
    }
    pub fn getMcast(self: *const Self) bool {
        return c.mlirCuteNVGPUCopyAtomBulkCopyG2STypeGetMcast(self.ptr());
    }
};

/// `!cute_nvgpu.atom.bulk_copy_s2g`: Bulk (TMA non-tensor) copy atom.
pub const CopyAtomBulkCopyS2GType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomBulkCopyS2G;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.bulk_copy_s2g";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int = 0,
        mask: bool = false,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomBulkCopyS2GTypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits, args.mask);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomBulkCopyS2GTypeGetValType(self.ptr()).ptr.?);
    }
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomBulkCopyS2GTypeGetCopyBits(self.ptr());
    }
    pub fn getMask(self: *const Self) bool {
        return c.mlirCuteNVGPUCopyAtomBulkCopyS2GTypeGetMask(self.ptr());
    }
};

/// `!cute_nvgpu.atom.bulk_copy_s2s`: Bulk (TMA non-tensor) copy atom.
pub const CopyAtomBulkCopyS2SType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomBulkCopyS2S;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.bulk_copy_s2s";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int = 0,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomBulkCopyS2STypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomBulkCopyS2STypeGetValType(self.ptr()).ptr.?);
    }
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomBulkCopyS2STypeGetCopyBits(self.ptr());
    }
};

/// `!cute_nvgpu.atom.dsmem_store`: Distributed shared-memory store atom.
pub const CopyAtomDsmemStoreType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomDsmemStore;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.dsmem_store";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomDsmemStoreTypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomDsmemStoreTypeGetValType(self.ptr()).ptr.?);
    }
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomDsmemStoreTypeGetCopyBits(self.ptr());
    }
};

/// `!cute_nvgpu.atom.g2r`: Global-to-register load atom.
pub const CopyAtomG2RType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomG2R;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.g2r";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int,
        memOrder: MemOrderKind = .WEAK,
        memScope: MemScopeKind = .CTA,
        l2PrefetchSize: L2PrefetchSize = .NONE,
        l1CacheEvictPriority: CacheEvictionPriority = .EVICT_NORMAL,
        loadCacheMode: LoadCacheMode = .always,
        sharedSpace: SharedSpace = .CTA,
        invariant: bool = false,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomG2RTypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits, @intFromEnum(args.memOrder), @intFromEnum(args.memScope), @intFromEnum(args.l2PrefetchSize), @intFromEnum(args.l1CacheEvictPriority), @intFromEnum(args.loadCacheMode), @intFromEnum(args.sharedSpace), args.invariant);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomG2RTypeGetValType(self.ptr()).ptr.?);
    }
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomG2RTypeGetCopyBits(self.ptr());
    }
    pub fn getMemOrder(self: *const Self) MemOrderKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomG2RTypeGetMemOrder(self.ptr()));
    }
    pub fn getMemScope(self: *const Self) MemScopeKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomG2RTypeGetMemScope(self.ptr()));
    }
    pub fn getL2PrefetchSize(self: *const Self) L2PrefetchSize {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomG2RTypeGetL2PrefetchSize(self.ptr()));
    }
    pub fn getL1CacheEvictPriority(self: *const Self) CacheEvictionPriority {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomG2RTypeGetL1CacheEvictPriority(self.ptr()));
    }
    pub fn getLoadCacheMode(self: *const Self) LoadCacheMode {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomG2RTypeGetLoadCacheMode(self.ptr()));
    }
    pub fn getSharedSpace(self: *const Self) SharedSpace {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomG2RTypeGetSharedSpace(self.ptr()));
    }
    pub fn getInvariant(self: *const Self) bool {
        return c.mlirCuteNVGPUCopyAtomG2RTypeGetInvariant(self.ptr());
    }
};

/// `!cute_nvgpu.atom.im2col_tma_load`: Executable im2col TMA load atom.
pub const CopyAtomIm2ColTmaLoadType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomIm2ColTmaLoad;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.im2col_tma_load";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int,
        numCta: c_int,
        gStride: *const mlir.Type,
        mcast: bool = false,
        tmaGbasis: ?*const mlir.Type = null,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomIm2ColTmaLoadTypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits, args.numCta, args.gStride.ptr(), args.mcast, if (args.tmaGbasis) |v| v.ptr() else c.MlirType{ .ptr = null });
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomIm2ColTmaLoadTypeGetValType(self.ptr()).ptr.?);
    }
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomIm2ColTmaLoadTypeGetCopyBits(self.ptr());
    }
    pub fn getNumCta(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomIm2ColTmaLoadTypeGetNumCta(self.ptr());
    }
    pub fn getGStride(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomIm2ColTmaLoadTypeGetGStride(self.ptr()).ptr.?);
    }
    pub fn getMcast(self: *const Self) bool {
        return c.mlirCuteNVGPUCopyAtomIm2ColTmaLoadTypeGetMcast(self.ptr());
    }
    pub fn getTmaGbasis(self: *const Self) ?*const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomIm2ColTmaLoadTypeGetTmaGbasis(self.ptr()).ptr);
    }
};

/// `!cute_nvgpu.atom.im2col_tma_store`: Executable im2col TMA store atom.
pub const CopyAtomIm2ColTmaStoreType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomIm2ColTmaStore;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.im2col_tma_store";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int,
        gStride: *const mlir.Type,
        tmaGbasis: ?*const mlir.Type = null,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomIm2ColTmaStoreTypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits, args.gStride.ptr(), if (args.tmaGbasis) |v| v.ptr() else c.MlirType{ .ptr = null });
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomIm2ColTmaStoreTypeGetValType(self.ptr()).ptr.?);
    }
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomIm2ColTmaStoreTypeGetCopyBits(self.ptr());
    }
    pub fn getGStride(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomIm2ColTmaStoreTypeGetGStride(self.ptr()).ptr.?);
    }
    pub fn getTmaGbasis(self: *const Self) ?*const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomIm2ColTmaStoreTypeGetTmaGbasis(self.ptr()).ptr);
    }
};

/// `!cute_nvgpu.atom.ldsm`: ldmatrix atom.
pub const CopyAtomLdsmType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomLdsm;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.ldsm";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        mode: *const mlir.Attribute,
        szPattern: LdsmSzPattern,
        numMatrices: c_int,
        transpose: bool = false,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomLdsmTypeGet(ctx.ptr(), args.valType.ptr(), args.mode.ptr(), @intFromEnum(args.szPattern), args.numMatrices, args.transpose);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomLdsmTypeGetValType(self.ptr()).ptr.?);
    }
    pub fn getMode(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomLdsmTypeGetMode(self.ptr()).ptr.?);
    }
    pub fn getSzPattern(self: *const Self) LdsmSzPattern {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomLdsmTypeGetSzPattern(self.ptr()));
    }
    pub fn getNumMatrices(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomLdsmTypeGetNumMatrices(self.ptr());
    }
    pub fn getTranspose(self: *const Self) bool {
        return c.mlirCuteNVGPUCopyAtomLdsmTypeGetTranspose(self.ptr());
    }
};

/// `!cute_nvgpu.atom.non_exec_2d_gather4_tma_load`: Non-executable 2-D gather4 TMA load atom.
pub const CopyAtomNonExec2DGather4TmaLoadType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomNonExec2DGather4TmaLoad;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.non_exec_2d_gather4_tma_load";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        kind: GatherScatterTmaLoad,
        valType: *const mlir.Type,
        copyBits: c_int,
        tmaGbasis: *const mlir.Type,
        tmaFormat: ?TmaDataFormat = null,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomNonExec2DGather4TmaLoadTypeGet(ctx.ptr(), @intFromEnum(args.kind), args.valType.ptr(), args.copyBits, args.tmaGbasis.ptr(), if (args.tmaFormat) |v| @intFromEnum(v) else -1);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getKind(self: *const Self) GatherScatterTmaLoad {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomNonExec2DGather4TmaLoadTypeGetKind(self.ptr()));
    }
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomNonExec2DGather4TmaLoadTypeGetValType(self.ptr()).ptr.?);
    }
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomNonExec2DGather4TmaLoadTypeGetCopyBits(self.ptr());
    }
    pub fn getTmaGbasis(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomNonExec2DGather4TmaLoadTypeGetTmaGbasis(self.ptr()).ptr.?);
    }
    pub fn getTmaFormat(self: *const Self) ?TmaDataFormat {
        const value = c.mlirCuteNVGPUCopyAtomNonExec2DGather4TmaLoadTypeGetTmaFormat(self.ptr());
        return if (value < 0) null else @enumFromInt(value);
    }
};

/// `!cute_nvgpu.atom.non_exec_2d_scatter4_tma_store`: Non-executable 2-D scatter4 TMA store atom.
pub const CopyAtomNonExec2DScatter4TmaStoreType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomNonExec2DScatter4TmaStore;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.non_exec_2d_scatter4_tma_store";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int,
        tmaGbasis: *const mlir.Type,
        tmaFormat: ?TmaDataFormat = null,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomNonExec2DScatter4TmaStoreTypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits, args.tmaGbasis.ptr(), if (args.tmaFormat) |v| @intFromEnum(v) else -1);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomNonExec2DScatter4TmaStoreTypeGetValType(self.ptr()).ptr.?);
    }
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomNonExec2DScatter4TmaStoreTypeGetCopyBits(self.ptr());
    }
    pub fn getTmaGbasis(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomNonExec2DScatter4TmaStoreTypeGetTmaGbasis(self.ptr()).ptr.?);
    }
    pub fn getTmaFormat(self: *const Self) ?TmaDataFormat {
        const value = c.mlirCuteNVGPUCopyAtomNonExec2DScatter4TmaStoreTypeGetTmaFormat(self.ptr());
        return if (value < 0) null else @enumFromInt(value);
    }
};

/// `!cute_nvgpu.atom.non_exec_im2col_tma_load`: Non-executable im2col TMA load atom.
pub const CopyAtomNonExecIm2ColTmaLoadType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomNonExecIm2ColTmaLoad;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.non_exec_im2col_tma_load";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        kind: Im2ColTmaLoad,
        valType: *const mlir.Type,
        copyBits: c_int,
        tmaGbasis: *const mlir.Type,
        tmaFormat: ?TmaDataFormat = null,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomNonExecIm2ColTmaLoadTypeGet(ctx.ptr(), @intFromEnum(args.kind), args.valType.ptr(), args.copyBits, args.tmaGbasis.ptr(), if (args.tmaFormat) |v| @intFromEnum(v) else -1);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getKind(self: *const Self) Im2ColTmaLoad {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomNonExecIm2ColTmaLoadTypeGetKind(self.ptr()));
    }
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomNonExecIm2ColTmaLoadTypeGetValType(self.ptr()).ptr.?);
    }
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomNonExecIm2ColTmaLoadTypeGetCopyBits(self.ptr());
    }
    pub fn getTmaGbasis(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomNonExecIm2ColTmaLoadTypeGetTmaGbasis(self.ptr()).ptr.?);
    }
    pub fn getTmaFormat(self: *const Self) ?TmaDataFormat {
        const value = c.mlirCuteNVGPUCopyAtomNonExecIm2ColTmaLoadTypeGetTmaFormat(self.ptr());
        return if (value < 0) null else @enumFromInt(value);
    }
};

/// `!cute_nvgpu.atom.non_exec_im2col_tma_store`: Non-executable im2col TMA store atom.
pub const CopyAtomNonExecIm2ColTmaStoreType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomNonExecIm2ColTmaStore;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.non_exec_im2col_tma_store";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int,
        tmaGbasis: *const mlir.Type,
        tmaFormat: ?TmaDataFormat = null,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomNonExecIm2ColTmaStoreTypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits, args.tmaGbasis.ptr(), if (args.tmaFormat) |v| @intFromEnum(v) else -1);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomNonExecIm2ColTmaStoreTypeGetValType(self.ptr()).ptr.?);
    }
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomNonExecIm2ColTmaStoreTypeGetCopyBits(self.ptr());
    }
    pub fn getTmaGbasis(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomNonExecIm2ColTmaStoreTypeGetTmaGbasis(self.ptr()).ptr.?);
    }
    pub fn getTmaFormat(self: *const Self) ?TmaDataFormat {
        const value = c.mlirCuteNVGPUCopyAtomNonExecIm2ColTmaStoreTypeGetTmaFormat(self.ptr());
        return if (value < 0) null else @enumFromInt(value);
    }
};

/// `!cute_nvgpu.atom.non_exec_tiled_tma_load`: Non-executable tiled TMA load atom.
pub const CopyAtomNonExecTiledTmaLoadType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomNonExecTiledTmaLoad;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.non_exec_tiled_tma_load";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        kind: TiledTmaLoad,
        valType: *const mlir.Type,
        copyBits: c_int,
        tmaGbasis: *const mlir.Type,
        tmaFormat: ?TmaDataFormat = null,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomNonExecTiledTmaLoadTypeGet(ctx.ptr(), @intFromEnum(args.kind), args.valType.ptr(), args.copyBits, args.tmaGbasis.ptr(), if (args.tmaFormat) |v| @intFromEnum(v) else -1);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getKind(self: *const Self) TiledTmaLoad {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomNonExecTiledTmaLoadTypeGetKind(self.ptr()));
    }
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomNonExecTiledTmaLoadTypeGetValType(self.ptr()).ptr.?);
    }
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomNonExecTiledTmaLoadTypeGetCopyBits(self.ptr());
    }
    pub fn getTmaGbasis(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomNonExecTiledTmaLoadTypeGetTmaGbasis(self.ptr()).ptr.?);
    }
    pub fn getTmaFormat(self: *const Self) ?TmaDataFormat {
        const value = c.mlirCuteNVGPUCopyAtomNonExecTiledTmaLoadTypeGetTmaFormat(self.ptr());
        return if (value < 0) null else @enumFromInt(value);
    }
};

/// `!cute_nvgpu.atom.non_exec_tiled_tma_reduce`: Non-executable tiled TMA reduce atom.
pub const CopyAtomNonExecTiledTmaReduceType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomNonExecTiledTmaReduce;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.non_exec_tiled_tma_reduce";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        kind: ReductionKind,
        valType: *const mlir.Type,
        copyBits: c_int,
        tmaGbasis: *const mlir.Type,
        tmaFormat: ?TmaDataFormat = null,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomNonExecTiledTmaReduceTypeGet(ctx.ptr(), @intFromEnum(args.kind), args.valType.ptr(), args.copyBits, args.tmaGbasis.ptr(), if (args.tmaFormat) |v| @intFromEnum(v) else -1);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getKind(self: *const Self) ReductionKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomNonExecTiledTmaReduceTypeGetKind(self.ptr()));
    }
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomNonExecTiledTmaReduceTypeGetValType(self.ptr()).ptr.?);
    }
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomNonExecTiledTmaReduceTypeGetCopyBits(self.ptr());
    }
    pub fn getTmaGbasis(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomNonExecTiledTmaReduceTypeGetTmaGbasis(self.ptr()).ptr.?);
    }
    pub fn getTmaFormat(self: *const Self) ?TmaDataFormat {
        const value = c.mlirCuteNVGPUCopyAtomNonExecTiledTmaReduceTypeGetTmaFormat(self.ptr());
        return if (value < 0) null else @enumFromInt(value);
    }
};

/// `!cute_nvgpu.atom.non_exec_tiled_tma_store`: Non-executable tiled TMA store atom.
pub const CopyAtomNonExecTiledTmaStoreType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomNonExecTiledTmaStore;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.non_exec_tiled_tma_store";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int,
        tmaGbasis: *const mlir.Type,
        tmaFormat: ?TmaDataFormat = null,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomNonExecTiledTmaStoreTypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits, args.tmaGbasis.ptr(), if (args.tmaFormat) |v| @intFromEnum(v) else -1);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomNonExecTiledTmaStoreTypeGetValType(self.ptr()).ptr.?);
    }
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomNonExecTiledTmaStoreTypeGetCopyBits(self.ptr());
    }
    pub fn getTmaGbasis(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomNonExecTiledTmaStoreTypeGetTmaGbasis(self.ptr()).ptr.?);
    }
    pub fn getTmaFormat(self: *const Self) ?TmaDataFormat {
        const value = c.mlirCuteNVGPUCopyAtomNonExecTiledTmaStoreTypeGetTmaFormat(self.ptr());
        return if (value < 0) null else @enumFromInt(value);
    }
};

/// `!cute_nvgpu.atom.r2g`: Register-to-global store atom.
pub const CopyAtomR2GType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomR2G;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.r2g";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int,
        memOrder: MemOrderKind = .WEAK,
        memScope: MemScopeKind = .CTA,
        l1CacheEvictPriority: CacheEvictionPriority = .EVICT_NORMAL,
        storeCacheMode: StoreCacheMode = .write_back,
        sharedSpace: SharedSpace = .CTA,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomR2GTypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits, @intFromEnum(args.memOrder), @intFromEnum(args.memScope), @intFromEnum(args.l1CacheEvictPriority), @intFromEnum(args.storeCacheMode), @intFromEnum(args.sharedSpace));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomR2GTypeGetValType(self.ptr()).ptr.?);
    }
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomR2GTypeGetCopyBits(self.ptr());
    }
    pub fn getMemOrder(self: *const Self) MemOrderKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomR2GTypeGetMemOrder(self.ptr()));
    }
    pub fn getMemScope(self: *const Self) MemScopeKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomR2GTypeGetMemScope(self.ptr()));
    }
    pub fn getL1CacheEvictPriority(self: *const Self) CacheEvictionPriority {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomR2GTypeGetL1CacheEvictPriority(self.ptr()));
    }
    pub fn getStoreCacheMode(self: *const Self) StoreCacheMode {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomR2GTypeGetStoreCacheMode(self.ptr()));
    }
    pub fn getSharedSpace(self: *const Self) SharedSpace {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomR2GTypeGetSharedSpace(self.ptr()));
    }
};

/// `!cute_nvgpu.atom.r2s`: Register-to-shared store atom.
pub const CopyAtomR2SType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomR2S;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.r2s";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int,
        memOrder: MemOrderKind = .WEAK,
        memScope: MemScopeKind = .CTA,
        sharedSpace: SharedSpace = .CTA,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomR2STypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits, @intFromEnum(args.memOrder), @intFromEnum(args.memScope), @intFromEnum(args.sharedSpace));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomR2STypeGetValType(self.ptr()).ptr.?);
    }
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomR2STypeGetCopyBits(self.ptr());
    }
    pub fn getMemOrder(self: *const Self) MemOrderKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomR2STypeGetMemOrder(self.ptr()));
    }
    pub fn getMemScope(self: *const Self) MemScopeKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomR2STypeGetMemScope(self.ptr()));
    }
    pub fn getSharedSpace(self: *const Self) SharedSpace {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomR2STypeGetSharedSpace(self.ptr()));
    }
};

/// `!cute_nvgpu.atom.s2r`: Shared-to-register load atom.
pub const CopyAtomS2RType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomS2R;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.s2r";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int,
        memOrder: MemOrderKind = .WEAK,
        memScope: MemScopeKind = .CTA,
        sharedSpace: SharedSpace = .CTA,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomS2RTypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits, @intFromEnum(args.memOrder), @intFromEnum(args.memScope), @intFromEnum(args.sharedSpace));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomS2RTypeGetValType(self.ptr()).ptr.?);
    }
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomS2RTypeGetCopyBits(self.ptr());
    }
    pub fn getMemOrder(self: *const Self) MemOrderKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomS2RTypeGetMemOrder(self.ptr()));
    }
    pub fn getMemScope(self: *const Self) MemScopeKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomS2RTypeGetMemScope(self.ptr()));
    }
    pub fn getSharedSpace(self: *const Self) SharedSpace {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomS2RTypeGetSharedSpace(self.ptr()));
    }
};

/// `!cute_nvgpu.atom.simt_async_copy`: cp.async copy atom.
pub const CopyAtomSIMTAsyncCopyType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomSIMTAsyncCopy;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.simt_async_copy";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        cache: LoadCacheMode,
        copyBits: c_int,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomSIMTAsyncCopyTypeGet(ctx.ptr(), args.valType.ptr(), @intFromEnum(args.cache), args.copyBits);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSIMTAsyncCopyTypeGetValType(self.ptr()).ptr.?);
    }
    pub fn getCache(self: *const Self) LoadCacheMode {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomSIMTAsyncCopyTypeGetCache(self.ptr()));
    }
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSIMTAsyncCopyTypeGetCopyBits(self.ptr());
    }
};

/// `!cute_nvgpu.atom.simt_multimem_ld_reduce`: multimem.ld_reduce atom.
pub const CopyAtomSIMTMultimemLdReduceType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomSIMTMultimemLdReduce;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.simt_multimem_ld_reduce";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int = 0,
        reduction: ReductionKind = .ADD,
        ldReduceAccPrecision: ?*const mlir.Attribute = null,
        memOrder: MemOrderKind = .RELAXED,
        memScope: MemScopeKind = .SYS,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomSIMTMultimemLdReduceTypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits, @intFromEnum(args.reduction), if (args.ldReduceAccPrecision) |v| v.ptr() else c.MlirAttribute{ .ptr = null }, @intFromEnum(args.memOrder), @intFromEnum(args.memScope));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSIMTMultimemLdReduceTypeGetValType(self.ptr()).ptr.?);
    }
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSIMTMultimemLdReduceTypeGetCopyBits(self.ptr());
    }
    pub fn getReduction(self: *const Self) ReductionKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomSIMTMultimemLdReduceTypeGetReduction(self.ptr()));
    }
    pub fn getLdReduceAccPrecision(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSIMTMultimemLdReduceTypeGetLdReduceAccPrecision(self.ptr()).ptr);
    }
    pub fn getMemOrder(self: *const Self) MemOrderKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomSIMTMultimemLdReduceTypeGetMemOrder(self.ptr()));
    }
    pub fn getMemScope(self: *const Self) MemScopeKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomSIMTMultimemLdReduceTypeGetMemScope(self.ptr()));
    }
};

/// `!cute_nvgpu.atom.simt_multimem_red`: multimem.red atom.
pub const CopyAtomSIMTMultimemRedType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomSIMTMultimemRed;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.simt_multimem_red";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int,
        reduction: ReductionKind = .ADD,
        memOrder: MemOrderKind = .RELAXED,
        memScope: MemScopeKind = .SYS,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomSIMTMultimemRedTypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits, @intFromEnum(args.reduction), @intFromEnum(args.memOrder), @intFromEnum(args.memScope));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSIMTMultimemRedTypeGetValType(self.ptr()).ptr.?);
    }
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSIMTMultimemRedTypeGetCopyBits(self.ptr());
    }
    pub fn getReduction(self: *const Self) ReductionKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomSIMTMultimemRedTypeGetReduction(self.ptr()));
    }
    pub fn getMemOrder(self: *const Self) MemOrderKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomSIMTMultimemRedTypeGetMemOrder(self.ptr()));
    }
    pub fn getMemScope(self: *const Self) MemScopeKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomSIMTMultimemRedTypeGetMemScope(self.ptr()));
    }
};

/// `!cute_nvgpu.atom.simt_multimem_st`: multimem.st atom.
pub const CopyAtomSIMTMultimemStType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomSIMTMultimemSt;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.simt_multimem_st";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int = 0,
        memOrder: MemOrderKind = .RELAXED,
        memScope: MemScopeKind = .SYS,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomSIMTMultimemStTypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits, @intFromEnum(args.memOrder), @intFromEnum(args.memScope));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSIMTMultimemStTypeGetValType(self.ptr()).ptr.?);
    }
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSIMTMultimemStTypeGetCopyBits(self.ptr());
    }
    pub fn getMemOrder(self: *const Self) MemOrderKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomSIMTMultimemStTypeGetMemOrder(self.ptr()));
    }
    pub fn getMemScope(self: *const Self) MemScopeKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomSIMTMultimemStTypeGetMemScope(self.ptr()));
    }
};

/// `!cute_nvgpu.atom.universal_copy`: Universal (SIMT) copy atom.
pub const CopyAtomSIMTSyncCopyType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomSIMTSyncCopy;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.universal_copy";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int = 0,
        srcSpace: cute.AddressSpace = .generic,
        dstSpace: cute.AddressSpace = .generic,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomSIMTSyncCopyTypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits, @intFromEnum(args.srcSpace), @intFromEnum(args.dstSpace));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSIMTSyncCopyTypeGetValType(self.ptr()).ptr.?);
    }
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSIMTSyncCopyTypeGetCopyBits(self.ptr());
    }
    pub fn getSrcSpace(self: *const Self) cute.AddressSpace {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomSIMTSyncCopyTypeGetSrcSpace(self.ptr()));
    }
    pub fn getDstSpace(self: *const Self) cute.AddressSpace {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomSIMTSyncCopyTypeGetDstSpace(self.ptr()));
    }
};

/// `!cute_nvgpu.atom.s2t_copy`: tcgen05.cp (shared to tensor memory) atom.
pub const CopyAtomSM100CopyS2TType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomSM100CopyS2T;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.s2t_copy";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        numDp: c_int,
        numBit: c_int,
        numCta: c_int,
        broadcast: CopyS2TBroadcast = .none,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomSM100CopyS2TTypeGet(ctx.ptr(), args.valType.ptr(), args.numDp, args.numBit, args.numCta, @intFromEnum(args.broadcast));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSM100CopyS2TTypeGetValType(self.ptr()).ptr.?);
    }
    pub fn getNumDp(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSM100CopyS2TTypeGetNumDp(self.ptr());
    }
    pub fn getNumBit(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSM100CopyS2TTypeGetNumBit(self.ptr());
    }
    pub fn getNumCta(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSM100CopyS2TTypeGetNumCta(self.ptr());
    }
    pub fn getBroadcast(self: *const Self) CopyS2TBroadcast {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomSM100CopyS2TTypeGetBroadcast(self.ptr()));
    }
};

/// `!cute_nvgpu.atom.sm100_s2t_copy_v2`: tcgen05.cp atom with the shared-memory major mode.
pub const CopyAtomSM100S2TCopyV2Type = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomSM100S2TCopyV2;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.sm100_s2t_copy_v2";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        numDp: c_int,
        numBit: c_int,
        numCta: c_int,
        smemMajor: cute.MajorMode,
        broadcast: CopyS2TBroadcast = .none,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomSM100S2TCopyV2TypeGet(ctx.ptr(), args.valType.ptr(), args.numDp, args.numBit, args.numCta, @intFromEnum(args.smemMajor), @intFromEnum(args.broadcast));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSM100S2TCopyV2TypeGetValType(self.ptr()).ptr.?);
    }
    pub fn getNumDp(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSM100S2TCopyV2TypeGetNumDp(self.ptr());
    }
    pub fn getNumBit(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSM100S2TCopyV2TypeGetNumBit(self.ptr());
    }
    pub fn getNumCta(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSM100S2TCopyV2TypeGetNumCta(self.ptr());
    }
    pub fn getSmemMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomSM100S2TCopyV2TypeGetSmemMajor(self.ptr()));
    }
    pub fn getBroadcast(self: *const Self) CopyS2TBroadcast {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomSM100S2TCopyV2TypeGetBroadcast(self.ptr()));
    }
};

/// `!cute_nvgpu.atom.tmem_load`: tcgen05.ld atom.
pub const CopyAtomSM100TmemLoadType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomSM100TmemLoad;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.tmem_load";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        numDp: c_int,
        numBit: c_int,
        numRep: c_uint,
        pack16b: bool = false,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomSM100TmemLoadTypeGet(ctx.ptr(), args.valType.ptr(), args.numDp, args.numBit, args.numRep, args.pack16b);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSM100TmemLoadTypeGetValType(self.ptr()).ptr.?);
    }
    pub fn getNumDp(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSM100TmemLoadTypeGetNumDp(self.ptr());
    }
    pub fn getNumBit(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSM100TmemLoadTypeGetNumBit(self.ptr());
    }
    pub fn getNumRep(self: *const Self) c_uint {
        return c.mlirCuteNVGPUCopyAtomSM100TmemLoadTypeGetNumRep(self.ptr());
    }
    pub fn getPack16b(self: *const Self) bool {
        return c.mlirCuteNVGPUCopyAtomSM100TmemLoadTypeGetPack16b(self.ptr());
    }
};

/// `!cute_nvgpu.atom.tmem_store`: tcgen05.st atom.
pub const CopyAtomSM100TmemStoreType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomSM100TmemStore;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.tmem_store";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        numDp: c_int,
        numBit: c_int,
        numRep: c_uint,
        expand16b: bool = false,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomSM100TmemStoreTypeGet(ctx.ptr(), args.valType.ptr(), args.numDp, args.numBit, args.numRep, args.expand16b);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSM100TmemStoreTypeGetValType(self.ptr()).ptr.?);
    }
    pub fn getNumDp(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSM100TmemStoreTypeGetNumDp(self.ptr());
    }
    pub fn getNumBit(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSM100TmemStoreTypeGetNumBit(self.ptr());
    }
    pub fn getNumRep(self: *const Self) c_uint {
        return c.mlirCuteNVGPUCopyAtomSM100TmemStoreTypeGetNumRep(self.ptr());
    }
    pub fn getExpand16b(self: *const Self) bool {
        return c.mlirCuteNVGPUCopyAtomSM100TmemStoreTypeGetExpand16b(self.ptr());
    }
};

/// `!cute_nvgpu.atom.tmem_load_spcompress`: SM107 tensor-memory load with sparse compression.
pub const CopyAtomSM107TmemLoadSPCompressType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomSM107TmemLoadSPCompress;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.tmem_load_spcompress";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        numDp: c_int,
        numBit: c_int,
        numRep: c_uint,
        redOp: TmemLoadRedOp,
        red: bool = false,
        nan: bool = false,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomSM107TmemLoadSPCompressTypeGet(ctx.ptr(), args.valType.ptr(), args.numDp, args.numBit, args.numRep, @intFromEnum(args.redOp), args.red, args.nan);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSM107TmemLoadSPCompressTypeGetValType(self.ptr()).ptr.?);
    }
    pub fn getNumDp(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSM107TmemLoadSPCompressTypeGetNumDp(self.ptr());
    }
    pub fn getNumBit(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSM107TmemLoadSPCompressTypeGetNumBit(self.ptr());
    }
    pub fn getNumRep(self: *const Self) c_uint {
        return c.mlirCuteNVGPUCopyAtomSM107TmemLoadSPCompressTypeGetNumRep(self.ptr());
    }
    pub fn getRedOp(self: *const Self) TmemLoadRedOp {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomSM107TmemLoadSPCompressTypeGetRedOp(self.ptr()));
    }
    pub fn getRed(self: *const Self) bool {
        return c.mlirCuteNVGPUCopyAtomSM107TmemLoadSPCompressTypeGetRed(self.ptr());
    }
    pub fn getNan(self: *const Self) bool {
        return c.mlirCuteNVGPUCopyAtomSM107TmemLoadSPCompressTypeGetNan(self.ptr());
    }
};

/// `!cute_nvgpu.atom.tmem_load_red`: tcgen05.ld.red atom.
pub const CopyAtomSM10xTmemLoadRedType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomSM10xTmemLoadRed;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.tmem_load_red";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        numDp: c_int,
        numBit: c_int,
        numRep: c_uint,
        redOp: TmemLoadRedOp,
        nan: bool = false,
        halfSplitOff: ?*const mlir.Attribute = null,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGet(ctx.ptr(), args.valType.ptr(), args.numDp, args.numBit, args.numRep, @intFromEnum(args.redOp), args.nan, if (args.halfSplitOff) |v| v.ptr() else c.MlirAttribute{ .ptr = null });
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGetValType(self.ptr()).ptr.?);
    }
    pub fn getNumDp(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGetNumDp(self.ptr());
    }
    pub fn getNumBit(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGetNumBit(self.ptr());
    }
    pub fn getNumRep(self: *const Self) c_uint {
        return c.mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGetNumRep(self.ptr());
    }
    pub fn getRedOp(self: *const Self) TmemLoadRedOp {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGetRedOp(self.ptr()));
    }
    pub fn getNan(self: *const Self) bool {
        return c.mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGetNan(self.ptr());
    }
    pub fn getHalfSplitOff(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGetHalfSplitOff(self.ptr()).ptr);
    }
};

/// `!cute_nvgpu.atom.stsm`: stmatrix atom.
pub const CopyAtomStsmType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomStsm;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.stsm";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        mode: *const mlir.Attribute,
        numMatrices: c_int,
        transpose: bool = false,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomStsmTypeGet(ctx.ptr(), args.valType.ptr(), args.mode.ptr(), args.numMatrices, args.transpose);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomStsmTypeGetValType(self.ptr()).ptr.?);
    }
    pub fn getMode(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomStsmTypeGetMode(self.ptr()).ptr.?);
    }
    pub fn getNumMatrices(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomStsmTypeGetNumMatrices(self.ptr());
    }
    pub fn getTranspose(self: *const Self) bool {
        return c.mlirCuteNVGPUCopyAtomStsmTypeGetTranspose(self.ptr());
    }
};

/// `!cute_nvgpu.atom.tma_load`: Executable TMA load atom.
pub const CopyAtomTmaLoadType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomTmaLoad;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.tma_load";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        sparsity: c_int = 1,
        copyBits: c_int,
        mode: TmaLoadMode,
        numCta: c_int,
        gStride: *const mlir.Type,
        mcast: bool = false,
        tmaGbasis: ?*const mlir.Type = null,
        override: bool = false,
        noFullyOobTile: bool = false,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomTmaLoadTypeGet(ctx.ptr(), args.valType.ptr(), args.sparsity, args.copyBits, @intFromEnum(args.mode), args.numCta, args.gStride.ptr(), args.mcast, if (args.tmaGbasis) |v| v.ptr() else c.MlirType{ .ptr = null }, args.override, args.noFullyOobTile);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomTmaLoadTypeGetValType(self.ptr()).ptr.?);
    }
    pub fn getSparsity(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomTmaLoadTypeGetSparsity(self.ptr());
    }
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomTmaLoadTypeGetCopyBits(self.ptr());
    }
    pub fn getMode(self: *const Self) TmaLoadMode {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomTmaLoadTypeGetMode(self.ptr()));
    }
    pub fn getNumCta(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomTmaLoadTypeGetNumCta(self.ptr());
    }
    pub fn getGStride(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomTmaLoadTypeGetGStride(self.ptr()).ptr.?);
    }
    pub fn getMcast(self: *const Self) bool {
        return c.mlirCuteNVGPUCopyAtomTmaLoadTypeGetMcast(self.ptr());
    }
    pub fn getTmaGbasis(self: *const Self) ?*const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomTmaLoadTypeGetTmaGbasis(self.ptr()).ptr);
    }
    pub fn getOverride(self: *const Self) bool {
        return c.mlirCuteNVGPUCopyAtomTmaLoadTypeGetOverride(self.ptr());
    }
    pub fn getNoFullyOobTile(self: *const Self) bool {
        return c.mlirCuteNVGPUCopyAtomTmaLoadTypeGetNoFullyOobTile(self.ptr());
    }
};

/// `!cute_nvgpu.atom.tma_reduce`: Executable TMA reduce atom.
pub const CopyAtomTmaReduceType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomTmaReduce;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.tma_reduce";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int,
        mode: TmaStoreMode,
        kind: ReductionKind,
        gStride: *const mlir.Type,
        tmaGbasis: ?*const mlir.Type = null,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomTmaReduceTypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits, @intFromEnum(args.mode), @intFromEnum(args.kind), args.gStride.ptr(), if (args.tmaGbasis) |v| v.ptr() else c.MlirType{ .ptr = null });
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomTmaReduceTypeGetValType(self.ptr()).ptr.?);
    }
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomTmaReduceTypeGetCopyBits(self.ptr());
    }
    pub fn getMode(self: *const Self) TmaStoreMode {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomTmaReduceTypeGetMode(self.ptr()));
    }
    pub fn getKind(self: *const Self) ReductionKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomTmaReduceTypeGetKind(self.ptr()));
    }
    pub fn getGStride(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomTmaReduceTypeGetGStride(self.ptr()).ptr.?);
    }
    pub fn getTmaGbasis(self: *const Self) ?*const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomTmaReduceTypeGetTmaGbasis(self.ptr()).ptr);
    }
};

/// `!cute_nvgpu.atom.tma_store`: Executable TMA store atom.
pub const CopyAtomTmaStoreType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomTmaStore;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.tma_store";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        sparsity: c_int = 1,
        copyBits: c_int,
        mode: TmaStoreMode,
        gStride: *const mlir.Type,
        tmaGbasis: ?*const mlir.Type = null,
        override: bool = false,
        noFullyOobTile: bool = false,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomTmaStoreTypeGet(ctx.ptr(), args.valType.ptr(), args.sparsity, args.copyBits, @intFromEnum(args.mode), args.gStride.ptr(), if (args.tmaGbasis) |v| v.ptr() else c.MlirType{ .ptr = null }, args.override, args.noFullyOobTile);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomTmaStoreTypeGetValType(self.ptr()).ptr.?);
    }
    pub fn getSparsity(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomTmaStoreTypeGetSparsity(self.ptr());
    }
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomTmaStoreTypeGetCopyBits(self.ptr());
    }
    pub fn getMode(self: *const Self) TmaStoreMode {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomTmaStoreTypeGetMode(self.ptr()));
    }
    pub fn getGStride(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomTmaStoreTypeGetGStride(self.ptr()).ptr.?);
    }
    pub fn getTmaGbasis(self: *const Self) ?*const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomTmaStoreTypeGetTmaGbasis(self.ptr()).ptr);
    }
    pub fn getOverride(self: *const Self) bool {
        return c.mlirCuteNVGPUCopyAtomTmaStoreTypeGetOverride(self.ptr());
    }
    pub fn getNoFullyOobTile(self: *const Self) bool {
        return c.mlirCuteNVGPUCopyAtomTmaStoreTypeGetNoFullyOobTile(self.ptr());
    }
};

/// `!cute_nvgpu.sm100.mma_bs_sp`: SM100 block-scaled sparse UMMA MMA atom.
pub const MmaAtomSM100UMMABlockScaledSparseType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM100UMMABlockScaledSparse;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sm100.mma_bs_sp";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        shapeMnk: *const mlir.Attribute,
        numCta: c_int,
        aMajor: cute.MajorMode,
        bMajor: cute.MajorMode,
        aType: *const mlir.Type,
        bType: *const mlir.Type,
        cType: *const mlir.Type,
        sfType: *const mlir.Type,
        sparseMetadataFormat: SparseMetadataFormat,
        aFragKind: MmaFragKind,
        vecSize: c_int,
        archPromote: Arch = .sm_100,
        intOverflow: ?*const mlir.Attribute = null,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGet(ctx.ptr(), args.shapeMnk.ptr(), args.numCta, @intFromEnum(args.aMajor), @intFromEnum(args.bMajor), args.aType.ptr(), args.bType.ptr(), args.cType.ptr(), args.sfType.ptr(), @intFromEnum(args.sparseMetadataFormat), @intFromEnum(args.aFragKind), args.vecSize, @intFromEnum(args.archPromote), if (args.intOverflow) |v| v.ptr() else c.MlirAttribute{ .ptr = null });
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getShapeMnk(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetShapeMnk(self.ptr()).ptr.?);
    }
    pub fn getNumCta(self: *const Self) c_int {
        return c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetNumCta(self.ptr());
    }
    pub fn getAMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetAMajor(self.ptr()));
    }
    pub fn getBMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetBMajor(self.ptr()));
    }
    pub fn getAType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetAType(self.ptr()).ptr.?);
    }
    pub fn getBType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetBType(self.ptr()).ptr.?);
    }
    pub fn getCType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetCType(self.ptr()).ptr.?);
    }
    pub fn getSfType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetSfType(self.ptr()).ptr.?);
    }
    pub fn getSparseMetadataFormat(self: *const Self) SparseMetadataFormat {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetSparseMetadataFormat(self.ptr()));
    }
    pub fn getAFragKind(self: *const Self) MmaFragKind {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetAFragKind(self.ptr()));
    }
    pub fn getVecSize(self: *const Self) c_int {
        return c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetVecSize(self.ptr());
    }
    pub fn getArchPromote(self: *const Self) Arch {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetArchPromote(self.ptr()));
    }
    pub fn getIntOverflow(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetIntOverflow(self.ptr()).ptr);
    }
};

/// `!cute_nvgpu.sm100.mma_bs`: SM100 block-scaled UMMA MMA atom.
pub const MmaAtomSM100UMMABlockScaledType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM100UMMABlockScaled;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sm100.mma_bs";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        shapeMnk: *const mlir.Attribute,
        numCta: c_int,
        aMajor: cute.MajorMode,
        bMajor: cute.MajorMode,
        aType: *const mlir.Type,
        bType: *const mlir.Type,
        cType: *const mlir.Type,
        sfType: *const mlir.Type,
        aFragKind: MmaFragKind,
        vecSize: c_int,
        archPromote: Arch = .sm_100,
        intOverflow: ?*const mlir.Attribute = null,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGet(ctx.ptr(), args.shapeMnk.ptr(), args.numCta, @intFromEnum(args.aMajor), @intFromEnum(args.bMajor), args.aType.ptr(), args.bType.ptr(), args.cType.ptr(), args.sfType.ptr(), @intFromEnum(args.aFragKind), args.vecSize, @intFromEnum(args.archPromote), if (args.intOverflow) |v| v.ptr() else c.MlirAttribute{ .ptr = null });
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getShapeMnk(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetShapeMnk(self.ptr()).ptr.?);
    }
    pub fn getNumCta(self: *const Self) c_int {
        return c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetNumCta(self.ptr());
    }
    pub fn getAMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetAMajor(self.ptr()));
    }
    pub fn getBMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetBMajor(self.ptr()));
    }
    pub fn getAType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetAType(self.ptr()).ptr.?);
    }
    pub fn getBType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetBType(self.ptr()).ptr.?);
    }
    pub fn getCType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetCType(self.ptr()).ptr.?);
    }
    pub fn getSfType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetSfType(self.ptr()).ptr.?);
    }
    pub fn getAFragKind(self: *const Self) MmaFragKind {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetAFragKind(self.ptr()));
    }
    pub fn getVecSize(self: *const Self) c_int {
        return c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetVecSize(self.ptr());
    }
    pub fn getArchPromote(self: *const Self) Arch {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetArchPromote(self.ptr()));
    }
    pub fn getIntOverflow(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetIntOverflow(self.ptr()).ptr);
    }
};

/// `!cute_nvgpu.sm100.mma_sp`: SM100 sparse UMMA MMA atom.
pub const MmaAtomSM100UMMASparseType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM100UMMASparse;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sm100.mma_sp";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        shapeMnk: *const mlir.Attribute,
        numCta: c_int,
        aMajor: cute.MajorMode,
        bMajor: cute.MajorMode,
        aType: *const mlir.Type,
        bType: *const mlir.Type,
        cType: *const mlir.Type,
        eType: *const mlir.Type,
        sparseMetadataFormat: SparseMetadataFormat,
        aFragKind: MmaFragKind,
        cScaleExp: c_int,
        intOverflow: ?*const mlir.Attribute = null,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGet(ctx.ptr(), args.shapeMnk.ptr(), args.numCta, @intFromEnum(args.aMajor), @intFromEnum(args.bMajor), args.aType.ptr(), args.bType.ptr(), args.cType.ptr(), args.eType.ptr(), @intFromEnum(args.sparseMetadataFormat), @intFromEnum(args.aFragKind), args.cScaleExp, if (args.intOverflow) |v| v.ptr() else c.MlirAttribute{ .ptr = null });
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getShapeMnk(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetShapeMnk(self.ptr()).ptr.?);
    }
    pub fn getNumCta(self: *const Self) c_int {
        return c.mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetNumCta(self.ptr());
    }
    pub fn getAMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetAMajor(self.ptr()));
    }
    pub fn getBMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetBMajor(self.ptr()));
    }
    pub fn getAType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetAType(self.ptr()).ptr.?);
    }
    pub fn getBType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetBType(self.ptr()).ptr.?);
    }
    pub fn getCType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetCType(self.ptr()).ptr.?);
    }
    pub fn getEType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetEType(self.ptr()).ptr.?);
    }
    pub fn getSparseMetadataFormat(self: *const Self) SparseMetadataFormat {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetSparseMetadataFormat(self.ptr()));
    }
    pub fn getAFragKind(self: *const Self) MmaFragKind {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetAFragKind(self.ptr()));
    }
    pub fn getCScaleExp(self: *const Self) c_int {
        return c.mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetCScaleExp(self.ptr());
    }
    pub fn getIntOverflow(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetIntOverflow(self.ptr()).ptr);
    }
};

/// `!cute_nvgpu.sm100.mma`: SM100 UMMA MMA atom.
pub const MmaAtomSM100UMMAType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM100UMMA;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sm100.mma";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        shapeMnk: *const mlir.Attribute,
        numCta: c_int,
        aMajor: cute.MajorMode,
        bMajor: cute.MajorMode,
        aType: *const mlir.Type,
        bType: *const mlir.Type,
        cType: *const mlir.Type,
        aFragKind: MmaFragKind,
        cScaleExp: c_int,
        intOverflow: ?*const mlir.Attribute = null,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM100UMMATypeGet(ctx.ptr(), args.shapeMnk.ptr(), args.numCta, @intFromEnum(args.aMajor), @intFromEnum(args.bMajor), args.aType.ptr(), args.bType.ptr(), args.cType.ptr(), @intFromEnum(args.aFragKind), args.cScaleExp, if (args.intOverflow) |v| v.ptr() else c.MlirAttribute{ .ptr = null });
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getShapeMnk(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMATypeGetShapeMnk(self.ptr()).ptr.?);
    }
    pub fn getNumCta(self: *const Self) c_int {
        return c.mlirCuteNVGPUMmaAtomSM100UMMATypeGetNumCta(self.ptr());
    }
    pub fn getAMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM100UMMATypeGetAMajor(self.ptr()));
    }
    pub fn getBMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM100UMMATypeGetBMajor(self.ptr()));
    }
    pub fn getAType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMATypeGetAType(self.ptr()).ptr.?);
    }
    pub fn getBType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMATypeGetBType(self.ptr()).ptr.?);
    }
    pub fn getCType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMATypeGetCType(self.ptr()).ptr.?);
    }
    pub fn getAFragKind(self: *const Self) MmaFragKind {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM100UMMATypeGetAFragKind(self.ptr()));
    }
    pub fn getCScaleExp(self: *const Self) c_int {
        return c.mlirCuteNVGPUMmaAtomSM100UMMATypeGetCScaleExp(self.ptr());
    }
    pub fn getIntOverflow(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMATypeGetIntOverflow(self.ptr()).ptr);
    }
};

/// `!cute_nvgpu.sm107.mma_bs_sp`: SM107 block-scaled sparse UMMA MMA atom.
pub const MmaAtomSM107UMMABlockScaledSparseType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM107UMMABlockScaledSparse;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sm107.mma_bs_sp";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        shapeMnk: *const mlir.Attribute,
        numCta: c_int,
        aMajor: cute.MajorMode,
        bMajor: cute.MajorMode,
        aType: *const mlir.Type,
        bType: *const mlir.Type,
        cType: *const mlir.Type,
        sfType: *const mlir.Type,
        sparseMetadataFormat: SparseMetadataFormat,
        aFragKind: MmaFragKind,
        vecSize: c_int,
        aCollectorOp: MmaCollectorOp,
        bCollectorOp: MmaCollectorOp,
        intOverflow: ?*const mlir.Attribute = null,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGet(ctx.ptr(), args.shapeMnk.ptr(), args.numCta, @intFromEnum(args.aMajor), @intFromEnum(args.bMajor), args.aType.ptr(), args.bType.ptr(), args.cType.ptr(), args.sfType.ptr(), @intFromEnum(args.sparseMetadataFormat), @intFromEnum(args.aFragKind), args.vecSize, @intFromEnum(args.aCollectorOp), @intFromEnum(args.bCollectorOp), if (args.intOverflow) |v| v.ptr() else c.MlirAttribute{ .ptr = null });
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getShapeMnk(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetShapeMnk(self.ptr()).ptr.?);
    }
    pub fn getNumCta(self: *const Self) c_int {
        return c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetNumCta(self.ptr());
    }
    pub fn getAMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetAMajor(self.ptr()));
    }
    pub fn getBMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetBMajor(self.ptr()));
    }
    pub fn getAType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetAType(self.ptr()).ptr.?);
    }
    pub fn getBType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetBType(self.ptr()).ptr.?);
    }
    pub fn getCType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetCType(self.ptr()).ptr.?);
    }
    pub fn getSfType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetSfType(self.ptr()).ptr.?);
    }
    pub fn getSparseMetadataFormat(self: *const Self) SparseMetadataFormat {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetSparseMetadataFormat(self.ptr()));
    }
    pub fn getAFragKind(self: *const Self) MmaFragKind {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetAFragKind(self.ptr()));
    }
    pub fn getVecSize(self: *const Self) c_int {
        return c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetVecSize(self.ptr());
    }
    pub fn getACollectorOp(self: *const Self) MmaCollectorOp {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetACollectorOp(self.ptr()));
    }
    pub fn getBCollectorOp(self: *const Self) MmaCollectorOp {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetBCollectorOp(self.ptr()));
    }
    pub fn getIntOverflow(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetIntOverflow(self.ptr()).ptr);
    }
};

/// `!cute_nvgpu.sm107.mma_bs`: SM107 block-scaled UMMA MMA atom.
pub const MmaAtomSM107UMMABlockScaledType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM107UMMABlockScaled;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sm107.mma_bs";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        shapeMnk: *const mlir.Attribute,
        numCta: c_int,
        aMajor: cute.MajorMode,
        bMajor: cute.MajorMode,
        aType: *const mlir.Type,
        bType: *const mlir.Type,
        cType: *const mlir.Type,
        sfType: *const mlir.Type,
        aFragKind: MmaFragKind,
        vecSize: c_int,
        aCollectorOp: MmaCollectorOp,
        bCollectorOp: MmaCollectorOp,
        intOverflow: ?*const mlir.Attribute = null,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGet(ctx.ptr(), args.shapeMnk.ptr(), args.numCta, @intFromEnum(args.aMajor), @intFromEnum(args.bMajor), args.aType.ptr(), args.bType.ptr(), args.cType.ptr(), args.sfType.ptr(), @intFromEnum(args.aFragKind), args.vecSize, @intFromEnum(args.aCollectorOp), @intFromEnum(args.bCollectorOp), if (args.intOverflow) |v| v.ptr() else c.MlirAttribute{ .ptr = null });
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getShapeMnk(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetShapeMnk(self.ptr()).ptr.?);
    }
    pub fn getNumCta(self: *const Self) c_int {
        return c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetNumCta(self.ptr());
    }
    pub fn getAMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetAMajor(self.ptr()));
    }
    pub fn getBMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetBMajor(self.ptr()));
    }
    pub fn getAType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetAType(self.ptr()).ptr.?);
    }
    pub fn getBType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetBType(self.ptr()).ptr.?);
    }
    pub fn getCType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetCType(self.ptr()).ptr.?);
    }
    pub fn getSfType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetSfType(self.ptr()).ptr.?);
    }
    pub fn getAFragKind(self: *const Self) MmaFragKind {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetAFragKind(self.ptr()));
    }
    pub fn getVecSize(self: *const Self) c_int {
        return c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetVecSize(self.ptr());
    }
    pub fn getACollectorOp(self: *const Self) MmaCollectorOp {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetACollectorOp(self.ptr()));
    }
    pub fn getBCollectorOp(self: *const Self) MmaCollectorOp {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetBCollectorOp(self.ptr()));
    }
    pub fn getIntOverflow(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetIntOverflow(self.ptr()).ptr);
    }
};

/// `!cute_nvgpu.sm107.mma_sp`: SM107 sparse UMMA MMA atom.
pub const MmaAtomSM107UMMASparseType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM107UMMASparse;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sm107.mma_sp";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        shapeMnk: *const mlir.Attribute,
        numCta: c_int,
        aMajor: cute.MajorMode,
        bMajor: cute.MajorMode,
        aType: *const mlir.Type,
        bType: *const mlir.Type,
        cType: *const mlir.Type,
        eType: *const mlir.Type,
        sparseMetadataFormat: SparseMetadataFormat,
        aFragKind: MmaFragKind,
        cScaleExp: c_int,
        aCollectorOp: MmaCollectorOp,
        bCollectorOp: MmaCollectorOp,
        intOverflow: ?*const mlir.Attribute = null,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGet(ctx.ptr(), args.shapeMnk.ptr(), args.numCta, @intFromEnum(args.aMajor), @intFromEnum(args.bMajor), args.aType.ptr(), args.bType.ptr(), args.cType.ptr(), args.eType.ptr(), @intFromEnum(args.sparseMetadataFormat), @intFromEnum(args.aFragKind), args.cScaleExp, @intFromEnum(args.aCollectorOp), @intFromEnum(args.bCollectorOp), if (args.intOverflow) |v| v.ptr() else c.MlirAttribute{ .ptr = null });
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getShapeMnk(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetShapeMnk(self.ptr()).ptr.?);
    }
    pub fn getNumCta(self: *const Self) c_int {
        return c.mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetNumCta(self.ptr());
    }
    pub fn getAMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetAMajor(self.ptr()));
    }
    pub fn getBMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetBMajor(self.ptr()));
    }
    pub fn getAType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetAType(self.ptr()).ptr.?);
    }
    pub fn getBType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetBType(self.ptr()).ptr.?);
    }
    pub fn getCType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetCType(self.ptr()).ptr.?);
    }
    pub fn getEType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetEType(self.ptr()).ptr.?);
    }
    pub fn getSparseMetadataFormat(self: *const Self) SparseMetadataFormat {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetSparseMetadataFormat(self.ptr()));
    }
    pub fn getAFragKind(self: *const Self) MmaFragKind {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetAFragKind(self.ptr()));
    }
    pub fn getCScaleExp(self: *const Self) c_int {
        return c.mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetCScaleExp(self.ptr());
    }
    pub fn getACollectorOp(self: *const Self) MmaCollectorOp {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetACollectorOp(self.ptr()));
    }
    pub fn getBCollectorOp(self: *const Self) MmaCollectorOp {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetBCollectorOp(self.ptr()));
    }
    pub fn getIntOverflow(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetIntOverflow(self.ptr()).ptr);
    }
};

/// `!cute_nvgpu.sm107.mma`: SM107 UMMA MMA atom.
pub const MmaAtomSM107UMMAType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM107UMMA;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sm107.mma";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        shapeMnk: *const mlir.Attribute,
        numCta: c_int,
        aMajor: cute.MajorMode,
        bMajor: cute.MajorMode,
        aType: *const mlir.Type,
        bType: *const mlir.Type,
        cType: *const mlir.Type,
        aFragKind: MmaFragKind,
        cScaleExp: c_int,
        aCollectorOp: MmaCollectorOp,
        bCollectorOp: MmaCollectorOp,
        intOverflow: ?*const mlir.Attribute = null,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM107UMMATypeGet(ctx.ptr(), args.shapeMnk.ptr(), args.numCta, @intFromEnum(args.aMajor), @intFromEnum(args.bMajor), args.aType.ptr(), args.bType.ptr(), args.cType.ptr(), @intFromEnum(args.aFragKind), args.cScaleExp, @intFromEnum(args.aCollectorOp), @intFromEnum(args.bCollectorOp), if (args.intOverflow) |v| v.ptr() else c.MlirAttribute{ .ptr = null });
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getShapeMnk(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMATypeGetShapeMnk(self.ptr()).ptr.?);
    }
    pub fn getNumCta(self: *const Self) c_int {
        return c.mlirCuteNVGPUMmaAtomSM107UMMATypeGetNumCta(self.ptr());
    }
    pub fn getAMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMATypeGetAMajor(self.ptr()));
    }
    pub fn getBMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMATypeGetBMajor(self.ptr()));
    }
    pub fn getAType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMATypeGetAType(self.ptr()).ptr.?);
    }
    pub fn getBType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMATypeGetBType(self.ptr()).ptr.?);
    }
    pub fn getCType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMATypeGetCType(self.ptr()).ptr.?);
    }
    pub fn getAFragKind(self: *const Self) MmaFragKind {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMATypeGetAFragKind(self.ptr()));
    }
    pub fn getCScaleExp(self: *const Self) c_int {
        return c.mlirCuteNVGPUMmaAtomSM107UMMATypeGetCScaleExp(self.ptr());
    }
    pub fn getACollectorOp(self: *const Self) MmaCollectorOp {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMATypeGetACollectorOp(self.ptr()));
    }
    pub fn getBCollectorOp(self: *const Self) MmaCollectorOp {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMATypeGetBCollectorOp(self.ptr()));
    }
    pub fn getIntOverflow(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMATypeGetIntOverflow(self.ptr()).ptr);
    }
};

/// `!cute_nvgpu.SM120.mma_bs`: SM120 block-scaled mma.sync MMA atom.
pub const MmaAtomSM120BlockScaledType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM120BlockScaled;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "SM120.mma_bs";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        shapeMnk: *const mlir.Attribute,
        vecSize: c_int,
        aType: *const mlir.Type,
        bType: *const mlir.Type,
        cType: *const mlir.Type,
        sfType: *const mlir.Type,
        useSfLayoutTV: bool,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGet(ctx.ptr(), args.shapeMnk.ptr(), args.vecSize, args.aType.ptr(), args.bType.ptr(), args.cType.ptr(), args.sfType.ptr(), args.useSfLayoutTV);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getShapeMnk(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGetShapeMnk(self.ptr()).ptr.?);
    }
    pub fn getVecSize(self: *const Self) c_int {
        return c.mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGetVecSize(self.ptr());
    }
    pub fn getAType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGetAType(self.ptr()).ptr.?);
    }
    pub fn getBType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGetBType(self.ptr()).ptr.?);
    }
    pub fn getCType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGetCType(self.ptr()).ptr.?);
    }
    pub fn getSfType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGetSfType(self.ptr()).ptr.?);
    }
    pub fn getUseSfLayoutTV(self: *const Self) bool {
        return c.mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGetUseSfLayoutTV(self.ptr());
    }
};

/// `!cute_nvgpu.sm80.sparse_mma`: SM80 sparse mma.sync MMA atom.
pub const MmaAtomSM80SparseType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM80Sparse;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sm80.sparse_mma";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        shapeMnk: *const mlir.Attribute,
        aType: *const mlir.Type,
        bType: *const mlir.Type,
        cType: *const mlir.Type,
        sparseMetadataFormat: SparseMetadataFormat,
        intOverflow: ?*const mlir.Attribute = null,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM80SparseTypeGet(ctx.ptr(), args.shapeMnk.ptr(), args.aType.ptr(), args.bType.ptr(), args.cType.ptr(), @intFromEnum(args.sparseMetadataFormat), if (args.intOverflow) |v| v.ptr() else c.MlirAttribute{ .ptr = null });
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getShapeMnk(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM80SparseTypeGetShapeMnk(self.ptr()).ptr.?);
    }
    pub fn getAType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM80SparseTypeGetAType(self.ptr()).ptr.?);
    }
    pub fn getBType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM80SparseTypeGetBType(self.ptr()).ptr.?);
    }
    pub fn getCType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM80SparseTypeGetCType(self.ptr()).ptr.?);
    }
    pub fn getSparseMetadataFormat(self: *const Self) SparseMetadataFormat {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM80SparseTypeGetSparseMetadataFormat(self.ptr()));
    }
    pub fn getIntOverflow(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM80SparseTypeGetIntOverflow(self.ptr()).ptr);
    }
};

/// `!cute_nvgpu.sm80.mma`: SM80 mma.sync MMA atom.
pub const MmaAtomSM80Type = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM80;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sm80.mma";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        shapeMnk: *const mlir.Attribute,
        aType: *const mlir.Type,
        bType: *const mlir.Type,
        cType: *const mlir.Type,
        intOverflow: ?*const mlir.Attribute = null,
        binaryOp: ?*const mlir.Attribute = null,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM80TypeGet(ctx.ptr(), args.shapeMnk.ptr(), args.aType.ptr(), args.bType.ptr(), args.cType.ptr(), if (args.intOverflow) |v| v.ptr() else c.MlirAttribute{ .ptr = null }, if (args.binaryOp) |v| v.ptr() else c.MlirAttribute{ .ptr = null });
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getShapeMnk(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM80TypeGetShapeMnk(self.ptr()).ptr.?);
    }
    pub fn getAType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM80TypeGetAType(self.ptr()).ptr.?);
    }
    pub fn getBType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM80TypeGetBType(self.ptr()).ptr.?);
    }
    pub fn getCType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM80TypeGetCType(self.ptr()).ptr.?);
    }
    pub fn getIntOverflow(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM80TypeGetIntOverflow(self.ptr()).ptr);
    }
    pub fn getBinaryOp(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM80TypeGetBinaryOp(self.ptr()).ptr);
    }
};

/// `!cute_nvgpu.sm89.mma`: SM89 fp8 mma.sync MMA atom.
pub const MmaAtomSM89Type = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM89;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sm89.mma";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        shapeMnk: *const mlir.Attribute,
        aType: *const mlir.Type,
        bType: *const mlir.Type,
        cType: *const mlir.Type,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM89TypeGet(ctx.ptr(), args.shapeMnk.ptr(), args.aType.ptr(), args.bType.ptr(), args.cType.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getShapeMnk(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM89TypeGetShapeMnk(self.ptr()).ptr.?);
    }
    pub fn getAType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM89TypeGetAType(self.ptr()).ptr.?);
    }
    pub fn getBType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM89TypeGetBType(self.ptr()).ptr.?);
    }
    pub fn getCType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM89TypeGetCType(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.sm90.mma`: SM90 wgmma MMA atom.
pub const MmaAtomSM90Type = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM90;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sm90.mma";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        shapeMnk: *const mlir.Attribute,
        aMajor: cute.MajorMode,
        bMajor: cute.MajorMode,
        aType: *const mlir.Type,
        bType: *const mlir.Type,
        cType: *const mlir.Type,
        aFragKind: MmaFragKind,
        intOverflow: ?*const mlir.Attribute = null,
        aNeg: bool = false,
        bNeg: bool = false,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM90TypeGet(ctx.ptr(), args.shapeMnk.ptr(), @intFromEnum(args.aMajor), @intFromEnum(args.bMajor), args.aType.ptr(), args.bType.ptr(), args.cType.ptr(), @intFromEnum(args.aFragKind), if (args.intOverflow) |v| v.ptr() else c.MlirAttribute{ .ptr = null }, args.aNeg, args.bNeg);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getShapeMnk(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM90TypeGetShapeMnk(self.ptr()).ptr.?);
    }
    pub fn getAMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM90TypeGetAMajor(self.ptr()));
    }
    pub fn getBMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM90TypeGetBMajor(self.ptr()));
    }
    pub fn getAType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM90TypeGetAType(self.ptr()).ptr.?);
    }
    pub fn getBType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM90TypeGetBType(self.ptr()).ptr.?);
    }
    pub fn getCType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM90TypeGetCType(self.ptr()).ptr.?);
    }
    pub fn getAFragKind(self: *const Self) MmaFragKind {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM90TypeGetAFragKind(self.ptr()));
    }
    pub fn getIntOverflow(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM90TypeGetIntOverflow(self.ptr()).ptr);
    }
    pub fn getANeg(self: *const Self) bool {
        return c.mlirCuteNVGPUMmaAtomSM90TypeGetANeg(self.ptr());
    }
    pub fn getBNeg(self: *const Self) bool {
        return c.mlirCuteNVGPUMmaAtomSM90TypeGetBNeg(self.ptr());
    }
};

/// `!cute_nvgpu.sm103.smem_desc_circular`: Circular shared-memory descriptor over a block layout in bytes.
pub const SmemDescCircularSM103Type = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUSmemDescCircularSM103;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sm103.smem_desc_circular";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        blockLayoutBytes: *const mlir.Attribute,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUSmemDescCircularSM103TypeGet(ctx.ptr(), args.blockLayoutBytes.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getBlockLayoutBytes(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUSmemDescCircularSM103TypeGetBlockLayoutBytes(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.sm107.smem_desc_circular`: Circular shared-memory descriptor over a block layout in bytes.
pub const SmemDescCircularSM107Type = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUSmemDescCircularSM107;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sm107.smem_desc_circular";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        blockLayoutBytes: *const mlir.Attribute,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUSmemDescCircularSM107TypeGet(ctx.ptr(), args.blockLayoutBytes.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getBlockLayoutBytes(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUSmemDescCircularSM107TypeGetBlockLayoutBytes(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.sm107.smem_desc`: SM107 shared-memory descriptor.
pub const SmemDescSM107Type = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUSmemDescSM107;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sm107.smem_desc";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCuteNVGPUSmemDescSM107TypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cute_nvgpu.smem_desc`: UMMA/GMMA shared-memory descriptor.
pub const SmemDescType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUSmemDesc;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "smem_desc";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCuteNVGPUSmemDescTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cute_nvgpu.smem_desc_view`: Shared-memory descriptor iterator with a layout.
pub const SmemDescViewType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUSmemDescView;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "smem_desc_view";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        desc: *const mlir.Type,
        layout: *const mlir.Attribute,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUSmemDescViewTypeGet(ctx.ptr(), args.desc.ptr(), args.layout.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getDesc(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUSmemDescViewTypeGetDesc(self.ptr()).ptr.?);
    }
    pub fn getLayout(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUSmemDescViewTypeGetLayout(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.tma_descriptor_im2col`: Im2col TMA descriptor.
pub const TmaDescriptorIm2ColType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUTmaDescriptorIm2Col;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tma_descriptor_im2col";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCuteNVGPUTmaDescriptorIm2ColTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cute_nvgpu.tma_descriptor_tiled`: Tiled TMA descriptor.
pub const TmaDescriptorTiledType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUTmaDescriptorTiled;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tma_descriptor_tiled";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCuteNVGPUTmaDescriptorTiledTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cute_nvgpu.atom.universal_fma`: Universal FMA MMA atom.
pub const UniversalFmaAtomType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUUniversalFmaAtom;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.universal_fma";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        shapeMnk: *const mlir.Attribute,
        aType: *const mlir.Type,
        bType: *const mlir.Type,
        cType: *const mlir.Type,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUUniversalFmaAtomTypeGet(ctx.ptr(), args.shapeMnk.ptr(), args.aType.ptr(), args.bType.ptr(), args.cType.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getShapeMnk(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUUniversalFmaAtomTypeGetShapeMnk(self.ptr()).ptr.?);
    }
    pub fn getAType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUUniversalFmaAtomTypeGetAType(self.ptr()).ptr.?);
    }
    pub fn getBType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUUniversalFmaAtomTypeGetBType(self.ptr()).ptr.?);
    }
    pub fn getCType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUUniversalFmaAtomTypeGetCType(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.workid_response`: Cluster launch control work-id response.
pub const WorkIdResponseType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUWorkIdResponse;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "workid_response";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCuteNVGPUWorkIdResponseTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `#cute_nvgpu.atom_copy_field_bulkg2s`: Fields stored in Bulk Load Copy Atom type.
pub const AtomCopyFieldBulkCopyG2SAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomCopyFieldBulkCopyG2S;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_copy_field_bulkg2s";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: AtomCopyFieldBulkCopyG2S,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomCopyFieldBulkCopyG2SAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) AtomCopyFieldBulkCopyG2S {
        return @enumFromInt(c.mlirCuteNVGPUAtomCopyFieldBulkCopyG2SAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.atom_copy_field_bulks2g`: Fields stored in Bulk Store Copy Atom type.
pub const AtomCopyFieldBulkCopyS2GAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomCopyFieldBulkCopyS2G;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_copy_field_bulks2g";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: AtomCopyFieldBulkCopyS2G,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomCopyFieldBulkCopyS2GAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) AtomCopyFieldBulkCopyS2G {
        return @enumFromInt(c.mlirCuteNVGPUAtomCopyFieldBulkCopyS2GAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.atom_copy_field_bulks2s`: Fields stored in Bulk CTA to Cluster Copy Atom type.
pub const AtomCopyFieldBulkCopyS2SAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomCopyFieldBulkCopyS2S;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_copy_field_bulks2s";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: AtomCopyFieldBulkCopyS2S,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomCopyFieldBulkCopyS2SAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) AtomCopyFieldBulkCopyS2S {
        return @enumFromInt(c.mlirCuteNVGPUAtomCopyFieldBulkCopyS2SAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.atom_copy_field_dsmem_store`: Fields stored in CopyAtomDsmemStoreType.
pub const AtomCopyFieldDsmemStoreAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomCopyFieldDsmemStore;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_copy_field_dsmem_store";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: AtomCopyFieldDsmemStore,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomCopyFieldDsmemStoreAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) AtomCopyFieldDsmemStore {
        return @enumFromInt(c.mlirCuteNVGPUAtomCopyFieldDsmemStoreAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.atom_copy_field_g2r`: Fields stored in CopyAtomG2RType.
pub const AtomCopyFieldLoadGlobalAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomCopyFieldLoadGlobal;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_copy_field_g2r";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: AtomCopyFieldLoadGlobal,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomCopyFieldLoadGlobalAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) AtomCopyFieldLoadGlobal {
        return @enumFromInt(c.mlirCuteNVGPUAtomCopyFieldLoadGlobalAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.atom_copy_field_non_exec_2d_gather4_tma_load`: Fields stored in CopyAtomNonExec2DGather4TmaLoadType.
pub const AtomCopyFieldNonExec2DGather4TmaLoadAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomCopyFieldNonExec2DGather4TmaLoad;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_copy_field_non_exec_2d_gather4_tma_load";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: AtomCopyFieldNonExec2DGather4TmaLoad,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomCopyFieldNonExec2DGather4TmaLoadAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) AtomCopyFieldNonExec2DGather4TmaLoad {
        return @enumFromInt(c.mlirCuteNVGPUAtomCopyFieldNonExec2DGather4TmaLoadAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.atom_copy_field_non_exec_2d_scatter4_tma_store`: Fields stored in CopyAtomNonExec2DScatter4TmaStoreType.
pub const AtomCopyFieldNonExec2DScatter4TmaStoreAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomCopyFieldNonExec2DScatter4TmaStore;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_copy_field_non_exec_2d_scatter4_tma_store";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: AtomCopyFieldNonExec2DScatter4TmaStore,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomCopyFieldNonExec2DScatter4TmaStoreAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) AtomCopyFieldNonExec2DScatter4TmaStore {
        return @enumFromInt(c.mlirCuteNVGPUAtomCopyFieldNonExec2DScatter4TmaStoreAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.atom_copy_field_non_exec_im2col_tma_load`: Fields stored in CopyAtomNonExecIm2ColTmaLoadType.
pub const AtomCopyFieldNonExecIm2ColTmaLoadAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomCopyFieldNonExecIm2ColTmaLoad;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_copy_field_non_exec_im2col_tma_load";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: AtomCopyFieldNonExecIm2ColTmaLoad,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomCopyFieldNonExecIm2ColTmaLoadAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) AtomCopyFieldNonExecIm2ColTmaLoad {
        return @enumFromInt(c.mlirCuteNVGPUAtomCopyFieldNonExecIm2ColTmaLoadAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.atom_copy_field_non_exec_im2col_tma_store`: Fields stored in CopyAtomNonExecIm2ColTmaStoreType.
pub const AtomCopyFieldNonExecIm2ColTmaStoreAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomCopyFieldNonExecIm2ColTmaStore;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_copy_field_non_exec_im2col_tma_store";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: AtomCopyFieldNonExecIm2ColTmaStore,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomCopyFieldNonExecIm2ColTmaStoreAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) AtomCopyFieldNonExecIm2ColTmaStore {
        return @enumFromInt(c.mlirCuteNVGPUAtomCopyFieldNonExecIm2ColTmaStoreAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.atom_copy_field_non_exec_tma_load`: Fields stored in CopyAtomNonExecTiledTmaLoadType.
pub const AtomCopyFieldNonExecTiledTmaLoadAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomCopyFieldNonExecTiledTmaLoad;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_copy_field_non_exec_tma_load";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: AtomCopyFieldNonExecTiledTmaLoad,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomCopyFieldNonExecTiledTmaLoadAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) AtomCopyFieldNonExecTiledTmaLoad {
        return @enumFromInt(c.mlirCuteNVGPUAtomCopyFieldNonExecTiledTmaLoadAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.atom_copy_field_non_exec_tma_reduce`: Fields stored in CopyAtomNonExecTiledTmaReduceType.
pub const AtomCopyFieldNonExecTiledTmaReduceAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomCopyFieldNonExecTiledTmaReduce;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_copy_field_non_exec_tma_reduce";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: AtomCopyFieldNonExecTiledTmaReduce,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomCopyFieldNonExecTiledTmaReduceAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) AtomCopyFieldNonExecTiledTmaReduce {
        return @enumFromInt(c.mlirCuteNVGPUAtomCopyFieldNonExecTiledTmaReduceAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.atom_copy_field_non_exec_tma_store`: Fields stored in CopyAtomNonExecTiledTmaStoreType.
pub const AtomCopyFieldNonExecTiledTmaStoreAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomCopyFieldNonExecTiledTmaStore;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_copy_field_non_exec_tma_store";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: AtomCopyFieldNonExecTiledTmaStore,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomCopyFieldNonExecTiledTmaStoreAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) AtomCopyFieldNonExecTiledTmaStore {
        return @enumFromInt(c.mlirCuteNVGPUAtomCopyFieldNonExecTiledTmaStoreAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.atom_copy_field_r2g`: Fields stored in CopyAtomR2GType.
pub const AtomCopyFieldStoreGlobalAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomCopyFieldStoreGlobal;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_copy_field_r2g";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: AtomCopyFieldStoreGlobal,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomCopyFieldStoreGlobalAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) AtomCopyFieldStoreGlobal {
        return @enumFromInt(c.mlirCuteNVGPUAtomCopyFieldStoreGlobalAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.atom_copy_field_tmaload`: Fields stored in Tma Load Copy Atom type.
pub const AtomCopyFieldTmaLoadAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomCopyFieldTmaLoad;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_copy_field_tmaload";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: AtomCopyFieldTmaLoad,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomCopyFieldTmaLoadAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) AtomCopyFieldTmaLoad {
        return @enumFromInt(c.mlirCuteNVGPUAtomCopyFieldTmaLoadAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.atom_copy_field_tmareduce`: Fields stored in Tma Reduce Copy Atom type.
pub const AtomCopyFieldTmaReduceAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomCopyFieldTmaReduce;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_copy_field_tmareduce";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: AtomCopyFieldTmaReduce,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomCopyFieldTmaReduceAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) AtomCopyFieldTmaReduce {
        return @enumFromInt(c.mlirCuteNVGPUAtomCopyFieldTmaReduceAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.atom_copy_field_tmastore`: Fields stored in Tma Store Copy Atom type.
pub const AtomCopyFieldTmaStoreAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomCopyFieldTmaStore;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_copy_field_tmastore";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: AtomCopyFieldTmaStore,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomCopyFieldTmaStoreAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) AtomCopyFieldTmaStore {
        return @enumFromInt(c.mlirCuteNVGPUAtomCopyFieldTmaStoreAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.atom_mma_field_sm100`: Fields stored in MMA Atom types for SM100.
pub const AtomMmaFieldSM100Attr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomMmaFieldSM100;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_mma_field_sm100";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: AtomMmaFieldSM100,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomMmaFieldSM100AttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) AtomMmaFieldSM100 {
        return @enumFromInt(c.mlirCuteNVGPUAtomMmaFieldSM100AttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.atom_mma_field_sm100_block_scaled`: Fields stored in MMA Atom types for SM100 (block scaled).
pub const AtomMmaFieldSM100BlockScaledAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomMmaFieldSM100BlockScaled;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_mma_field_sm100_block_scaled";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: AtomMmaFieldSM100BlockScaled,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomMmaFieldSM100BlockScaledAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) AtomMmaFieldSM100BlockScaled {
        return @enumFromInt(c.mlirCuteNVGPUAtomMmaFieldSM100BlockScaledAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.atom_mma_field_sm100_block_scaled_sparse`: Fields stored in MMA Atom types for SM100 (block scaled).
pub const AtomMmaFieldSM100BlockScaledSparseAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomMmaFieldSM100BlockScaledSparse;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_mma_field_sm100_block_scaled_sparse";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: AtomMmaFieldSM100BlockScaledSparse,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomMmaFieldSM100BlockScaledSparseAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) AtomMmaFieldSM100BlockScaledSparse {
        return @enumFromInt(c.mlirCuteNVGPUAtomMmaFieldSM100BlockScaledSparseAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.atom_mma_field_sm100_sparse`: Fields stored in sparse MMA Atom types for SM100.
pub const AtomMmaFieldSM100SparseAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomMmaFieldSM100Sparse;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_mma_field_sm100_sparse";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: AtomMmaFieldSM100Sparse,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomMmaFieldSM100SparseAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) AtomMmaFieldSM100Sparse {
        return @enumFromInt(c.mlirCuteNVGPUAtomMmaFieldSM100SparseAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.atom_mma_field_sm120_block_scaled`: Fields stored in MMA Atom types for SM120 (block scaled).
pub const AtomMmaFieldSM120BlockScaledAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomMmaFieldSM120BlockScaled;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_mma_field_sm120_block_scaled";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: AtomMmaFieldSM120BlockScaled,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomMmaFieldSM120BlockScaledAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) AtomMmaFieldSM120BlockScaled {
        return @enumFromInt(c.mlirCuteNVGPUAtomMmaFieldSM120BlockScaledAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.atom_mma_field_sm80_sparse`: Fields stored in MMA Atom types for SM80.
pub const AtomMmaFieldSM80SparseAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomMmaFieldSM80Sparse;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_mma_field_sm80_sparse";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: AtomMmaFieldSM80Sparse,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomMmaFieldSM80SparseAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) AtomMmaFieldSM80Sparse {
        return @enumFromInt(c.mlirCuteNVGPUAtomMmaFieldSM80SparseAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.atom_mma_field_sm90`: Fields stored in MMA Atom types for SM90.
pub const AtomMmaFieldSM90Attr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomMmaFieldSM90;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_mma_field_sm90";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: AtomMmaFieldSM90,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomMmaFieldSM90AttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) AtomMmaFieldSM90 {
        return @enumFromInt(c.mlirCuteNVGPUAtomMmaFieldSM90AttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.bin_op`: Binary operation for single-bit MMA operations.
pub const BinaryOpAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUBinaryOp;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "bin_op";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: BinaryOp,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUBinaryOpAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) BinaryOp {
        return @enumFromInt(c.mlirCuteNVGPUBinaryOpAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.copy_s2t_broadcast_mode`: Broadcast modes for the different utccp instructions.
pub const CopyS2TBroadcastAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUCopyS2TBroadcast;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "copy_s2t_broadcast_mode";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: CopyS2TBroadcast,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyS2TBroadcastAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) CopyS2TBroadcast {
        return @enumFromInt(c.mlirCuteNVGPUCopyS2TBroadcastAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.gather_scatter_tma_load`: The various kinds of TMA loads in gather/scatter mode.
pub const GatherScatterTmaLoadAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUGatherScatterTmaLoad;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "gather_scatter_tma_load";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: GatherScatterTmaLoad,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUGatherScatterTmaLoadAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) GatherScatterTmaLoad {
        return @enumFromInt(c.mlirCuteNVGPUGatherScatterTmaLoadAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.im2col_tma_load`: The various kinds of TMA loads in im2col mode.
pub const Im2ColTmaLoadAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUIm2ColTmaLoad;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "im2col_tma_load";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: Im2ColTmaLoad,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUIm2ColTmaLoadAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) Im2ColTmaLoad {
        return @enumFromInt(c.mlirCuteNVGPUIm2ColTmaLoadAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.ld_reduce_acc_precision_kind`: multimem ld_reduce accumulation precision kind.
pub const LdReduceAccPrecisionKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPULdReduceAccPrecisionKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "ld_reduce_acc_precision_kind";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: LdReduceAccPrecisionKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPULdReduceAccPrecisionKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) LdReduceAccPrecisionKind {
        return @enumFromInt(c.mlirCuteNVGPULdReduceAccPrecisionKindAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.ldsm_sz_pattern`: LDSM's sz pattern, describing the bit size.
pub const LdsmSzPatternAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPULdsmSzPattern;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "ldsm_sz_pattern";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: LdsmSzPattern,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPULdsmSzPatternAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) LdsmSzPattern {
        return @enumFromInt(c.mlirCuteNVGPULdsmSzPatternAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.load_cache_mode`: Cache modes for the load instructions.
pub const LoadCacheModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPULoadCacheMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "load_cache_mode";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: LoadCacheMode,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPULoadCacheModeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) LoadCacheMode {
        return @enumFromInt(c.mlirCuteNVGPULoadCacheModeAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.mma_int_overflow`: MMA overflow options.
pub const MMAIntOverflowAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUMMAIntOverflow;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "mma_int_overflow";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MMAIntOverflow,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMMAIntOverflowAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) MMAIntOverflow {
        return @enumFromInt(c.mlirCuteNVGPUMMAIntOverflowAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.major`: Major mode for MMA operations.
pub const MajorModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUMajorMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "major";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: cute.MajorMode,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMajorModeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMajorModeAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.mma_collector_op`: Enums for the mma collector op.
pub const MmaCollectorOpAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUMmaCollectorOp;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "mma_collector_op";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MmaCollectorOp,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaCollectorOpAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) MmaCollectorOp {
        return @enumFromInt(c.mlirCuteNVGPUMmaCollectorOpAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.mma_frag_kind`: Enums for the mma frag type.
pub const MmaFragKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUMmaFragKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "mma_frag_kind";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MmaFragKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaFragKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) MmaFragKind {
        return @enumFromInt(c.mlirCuteNVGPUMmaFragKindAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.not_implemented_frg`: Fragment of an atom that has none.
pub const NotImplementedFrgAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUNotImplementedFrg;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "not_implemented_frg";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCuteNVGPUNotImplementedFrgAttrGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `#cute_nvgpu.tma_reduce_kind`: Op for the TMASTORE instruction.
pub const ReductionKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUReductionKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tma_reduce_kind";
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
        const result = c.mlirCuteNVGPUReductionKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) ReductionKind {
        return @enumFromInt(c.mlirCuteNVGPUReductionKindAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.rmem_frg`: Register-memory fragment of an MMA operand.
pub const RmemFrgAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPURmemFrg;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "rmem_frg";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        valueType: *const mlir.Type,
        operand: cute.MmaOperand,
        deriveElemType: bool = false,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPURmemFrgAttrGet(ctx.ptr(), args.valueType.ptr(), @intFromEnum(args.operand), args.deriveElemType);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValueType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPURmemFrgAttrGetValueType(self.ptr()).ptr.?);
    }
    pub fn getOperand(self: *const Self) cute.MmaOperand {
        return @enumFromInt(c.mlirCuteNVGPURmemFrgAttrGetOperand(self.ptr()));
    }
    pub fn getDeriveElemType(self: *const Self) bool {
        return c.mlirCuteNVGPURmemFrgAttrGetDeriveElemType(self.ptr());
    }
};

/// `#cute_nvgpu.arch.sm100.circular_smem_frg`: SM100 circular shared-memory fragment.
pub const SM100CircularSmemFrgAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUSM100CircularSmemFrg;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "arch.sm100.circular_smem_frg";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        majorMode: cute.MajorMode,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUSM100CircularSmemFrgAttrGet(ctx.ptr(), @intFromEnum(args.majorMode));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getMajorMode(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUSM100CircularSmemFrgAttrGetMajorMode(self.ptr()));
    }
};

/// `#cute_nvgpu.arch.sm100.smem_frg`: SM100 shared-memory fragment.
pub const SM100SmemFrgAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUSM100SmemFrg;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "arch.sm100.smem_frg";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        majorMode: cute.MajorMode,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUSM100SmemFrgAttrGet(ctx.ptr(), @intFromEnum(args.majorMode));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getMajorMode(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUSM100SmemFrgAttrGetMajorMode(self.ptr()));
    }
};

/// `#cute_nvgpu.arch.sm100.tmem_e_frg`: SM100 tensor-memory sparse-metadata fragment.
pub const SM100TmemEFrgAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUSM100TmemEFrg;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "arch.sm100.tmem_e_frg";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        aType: *const mlir.Type,
        eType: *const mlir.Type,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUSM100TmemEFrgAttrGet(ctx.ptr(), args.aType.ptr(), args.eType.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getAType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUSM100TmemEFrgAttrGetAType(self.ptr()).ptr.?);
    }
    pub fn getEType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUSM100TmemEFrgAttrGetEType(self.ptr()).ptr.?);
    }
};

/// `#cute_nvgpu.arch.sm100.tmem_frg`: SM100 tensor-memory fragment.
pub const SM100TmemFrgAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUSM100TmemFrg;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "arch.sm100.tmem_frg";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        dataType: *const mlir.Type,
        storageType: *const mlir.Type,
        ctaGroup: c_int,
        tmemAllocMode: TmemAllocMode,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUSM100TmemFrgAttrGet(ctx.ptr(), args.dataType.ptr(), args.storageType.ptr(), args.ctaGroup, @intFromEnum(args.tmemAllocMode));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getDataType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUSM100TmemFrgAttrGetDataType(self.ptr()).ptr.?);
    }
    pub fn getStorageType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUSM100TmemFrgAttrGetStorageType(self.ptr()).ptr.?);
    }
    pub fn getCtaGroup(self: *const Self) c_int {
        return c.mlirCuteNVGPUSM100TmemFrgAttrGetCtaGroup(self.ptr());
    }
    pub fn getTmemAllocMode(self: *const Self) TmemAllocMode {
        return @enumFromInt(c.mlirCuteNVGPUSM100TmemFrgAttrGetTmemAllocMode(self.ptr()));
    }
};

/// `#cute_nvgpu.arch.sm100.tmem_sf_frg`: SM100 tensor-memory scale-factor fragment.
pub const SM100TmemSfFrgAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUSM100TmemSfFrg;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "arch.sm100.tmem_sf_frg";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        sfType: *const mlir.Type,
        sfVecSize: c_int,
        ctaGroup: c_int,
        isSfa: bool,
        tmemAllocMode: TmemAllocMode,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUSM100TmemSfFrgAttrGet(ctx.ptr(), args.sfType.ptr(), args.sfVecSize, args.ctaGroup, args.isSfa, @intFromEnum(args.tmemAllocMode));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getSfType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUSM100TmemSfFrgAttrGetSfType(self.ptr()).ptr.?);
    }
    pub fn getSfVecSize(self: *const Self) c_int {
        return c.mlirCuteNVGPUSM100TmemSfFrgAttrGetSfVecSize(self.ptr());
    }
    pub fn getCtaGroup(self: *const Self) c_int {
        return c.mlirCuteNVGPUSM100TmemSfFrgAttrGetCtaGroup(self.ptr());
    }
    pub fn getIsSfa(self: *const Self) bool {
        return c.mlirCuteNVGPUSM100TmemSfFrgAttrGetIsSfa(self.ptr());
    }
    pub fn getTmemAllocMode(self: *const Self) TmemAllocMode {
        return @enumFromInt(c.mlirCuteNVGPUSM100TmemSfFrgAttrGetTmemAllocMode(self.ptr()));
    }
};

/// `#cute_nvgpu.arch.sm107.smem_frg`: SM107 shared-memory fragment.
pub const SM107SmemFrgAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUSM107SmemFrg;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "arch.sm107.smem_frg";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        majorMode: cute.MajorMode,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUSM107SmemFrgAttrGet(ctx.ptr(), @intFromEnum(args.majorMode));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getMajorMode(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUSM107SmemFrgAttrGetMajorMode(self.ptr()));
    }
};

/// `#cute_nvgpu.arch.sm90.smem_frg`: SM90 shared-memory fragment.
pub const SM90SmemFrgAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUSM90SmemFrg;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "arch.sm90.smem_frg";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        majorMode: cute.MajorMode,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUSM90SmemFrgAttrGet(ctx.ptr(), @intFromEnum(args.majorMode));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getMajorMode(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUSM90SmemFrgAttrGetMajorMode(self.ptr()));
    }
};

/// `#cute_nvgpu.tiled_tma_load`: The various kinds of TMA loads in tiled mode.
pub const TiledTmaLoadAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUTiledTmaLoad;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tiled_tma_load";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: TiledTmaLoad,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUTiledTmaLoadAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) TiledTmaLoad {
        return @enumFromInt(c.mlirCuteNVGPUTiledTmaLoadAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.tma_data_format`: Bits 74-71 of the TMA descriptor.
pub const TmaDataFormatAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUTmaDataFormat;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tma_data_format";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: TmaDataFormat,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUTmaDataFormatAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) TmaDataFormat {
        return @enumFromInt(c.mlirCuteNVGPUTmaDataFormatAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.tma_load_mode`: Modes for the TMA load instruction.
pub const TmaLoadModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUTmaLoadMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tma_load_mode";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: TmaLoadMode,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUTmaLoadModeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) TmaLoadMode {
        return @enumFromInt(c.mlirCuteNVGPUTmaLoadModeAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.tma_store_mode`: Modes for the TMA store instruction.
pub const TmaStoreModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUTmaStoreMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tma_store_mode";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: TmaStoreMode,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUTmaStoreModeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) TmaStoreMode {
        return @enumFromInt(c.mlirCuteNVGPUTmaStoreModeAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.tmem_alloc_mode`: Modes for the Tmem allocation.
pub const TmemAllocModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUTmemAllocMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tmem_alloc_mode";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: TmemAllocMode,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUTmemAllocModeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) TmemAllocMode {
        return @enumFromInt(c.mlirCuteNVGPUTmemAllocModeAttrGetValue(self.ptr()));
    }
};

/// `#cute_nvgpu.tmem_load_red_op`: Reduce operation for the Tmem load instruction.
pub const TmemLoadRedOpAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUTmemLoadRedOp;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tmem_load_red_op";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: TmemLoadRedOp,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUTmemLoadRedOpAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) TmemLoadRedOp {
        return @enumFromInt(c.mlirCuteNVGPUTmemLoadRedOpAttrGetValue(self.ptr()));
    }
};

// Operation builders, generated from the operations' .td.
/// `cute_nvgpu.arch.alloc_rmem`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_alloc_rmem(ctx: *mlir.Context, input: *const mlir.Value, ptr_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.alloc_rmem", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{ptr_type} },
        .location = location,
    });
}

/// `cute_nvgpu.arch.alloc_smem`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_alloc_smem(ctx: *mlir.Context, ptr_type: *const mlir.Type, input: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.alloc_smem", .{
        .results = .{ .flat = &.{ptr_type} },
        .attributes = &.{
            .named(ctx, "input", input),
        },
        .location = location,
    });
}

/// `cute_nvgpu.arch.copy.SM100.copy_s2t`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_copy_SM100_copy_s2t(ctx: *mlir.Context, src: *const mlir.Value, dst: *const mlir.Value, dp: *const mlir.Attribute, bits: *const mlir.Attribute, cta: *const mlir.Attribute, broadcast: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.copy.SM100.copy_s2t", .{
        .operands = .{ .flat = &.{ src, dst } },
        .attributes = &.{
            .named(ctx, "dp", dp),
            .named(ctx, "bits", bits),
            .named(ctx, "cta", cta),
            .named(ctx, "broadcast", broadcast),
        },
        .location = location,
    });
}

/// `cute_nvgpu.arch.copy.SM100.tma_load`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_copy_SM100_tma_load(ctx: *mlir.Context, src_desc: *const mlir.Value, dsmem_data_addr: *const mlir.Value, dsmem_bar_addr: *const mlir.Value, coord: []const *const mlir.Value, multicast_mask: ?*const mlir.Value, offsets: []const *const mlir.Value, cache_policy: ?*const mlir.Value, mode: *const mlir.Attribute, num_cta: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.copy.SM100.tma_load", .{
        .operands = .{ .variadic = &.{
            &.{src_desc},
            &.{dsmem_data_addr},
            &.{dsmem_bar_addr},
            coord,
            if (multicast_mask) |value| &.{value} else &.{},
            offsets,
            if (cache_policy) |value| &.{value} else &.{},
        } },
        .attributes = &.{
            .named(ctx, "mode", mode),
            .named(ctx, "num_cta", num_cta),
        },
        .location = location,
    });
}

/// `cute_nvgpu.arch.copy.SM100.tma_reduce`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_copy_SM100_tma_reduce(ctx: *mlir.Context, dst_desc: *const mlir.Value, src_smem_data_addr: *const mlir.Value, coord: []const *const mlir.Value, cache_policy: ?*const mlir.Value, mode: *const mlir.Attribute, kind: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.copy.SM100.tma_reduce", .{
        .operands = .{ .variadic = &.{
            &.{dst_desc},
            &.{src_smem_data_addr},
            coord,
            if (cache_policy) |value| &.{value} else &.{},
        } },
        .attributes = &.{
            .named(ctx, "mode", mode),
            .named(ctx, "kind", kind),
        },
        .location = location,
    });
}

/// `cute_nvgpu.arch.copy.SM100.tma_store`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_copy_SM100_tma_store(ctx: *mlir.Context, dst_desc: *const mlir.Value, src_smem_data_addr: *const mlir.Value, coord: []const *const mlir.Value, cache_policy: ?*const mlir.Value, mode: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.copy.SM100.tma_store", .{
        .operands = .{ .variadic = &.{
            &.{dst_desc},
            &.{src_smem_data_addr},
            coord,
            if (cache_policy) |value| &.{value} else &.{},
        } },
        .attributes = &.{
            .named(ctx, "mode", mode),
        },
        .location = location,
    });
}

/// `cute_nvgpu.arch.copy.SM100.tmem_load`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_copy_SM100_tmem_load(ctx: *mlir.Context, src: *const mlir.Value, dst_type: *const mlir.Type, num_dp: *const mlir.Attribute, num_b: *const mlir.Attribute, num_rep: *const mlir.Attribute, pack_16: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "num_dp", num_dp));
    attributes.appendAssumeCapacity(.named(ctx, "num_b", num_b));
    attributes.appendAssumeCapacity(.named(ctx, "num_rep", num_rep));
    if (pack_16) |value| attributes.appendAssumeCapacity(.named(ctx, "pack_16", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.copy.SM100.tmem_load", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute_nvgpu.arch.copy.SM100.tmem_store`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_copy_SM100_tmem_store(ctx: *mlir.Context, dst: *const mlir.Value, src: *const mlir.Value, num_dp: *const mlir.Attribute, num_b: *const mlir.Attribute, num_rep: *const mlir.Attribute, expand_16: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "num_dp", num_dp));
    attributes.appendAssumeCapacity(.named(ctx, "num_b", num_b));
    attributes.appendAssumeCapacity(.named(ctx, "num_rep", num_rep));
    if (expand_16) |value| attributes.appendAssumeCapacity(.named(ctx, "expand_16", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.copy.SM100.tmem_store", .{
        .operands = .{ .flat = &.{ dst, src } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute_nvgpu.arch.copy.SM107.tma_load_override`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_copy_SM107_tma_load_override(ctx: *mlir.Context, src_desc: *const mlir.Value, dsmem_data_addr: *const mlir.Value, dsmem_bar_addr: *const mlir.Value, override_addr: *const mlir.Value, tensor_size: []const *const mlir.Value, lower_stride: []const *const mlir.Value, upper_stride: ?*const mlir.Value, coord: []const *const mlir.Value, multicast_mask: ?*const mlir.Value, cache_policy: ?*const mlir.Value, mode: *const mlir.Attribute, num_cta: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.copy.SM107.tma_load_override", .{
        .operands = .{ .variadic = &.{
            &.{src_desc},
            &.{dsmem_data_addr},
            &.{dsmem_bar_addr},
            &.{override_addr},
            tensor_size,
            lower_stride,
            if (upper_stride) |value| &.{value} else &.{},
            coord,
            if (multicast_mask) |value| &.{value} else &.{},
            if (cache_policy) |value| &.{value} else &.{},
        } },
        .attributes = &.{
            .named(ctx, "mode", mode),
            .named(ctx, "num_cta", num_cta),
        },
        .location = location,
    });
}

/// `cute_nvgpu.arch.copy.SM107.tma_store_override`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_copy_SM107_tma_store_override(ctx: *mlir.Context, dst_desc: *const mlir.Value, src_smem_data_addr: *const mlir.Value, override_addr: *const mlir.Value, tensor_size: []const *const mlir.Value, lower_stride: []const *const mlir.Value, upper_stride: ?*const mlir.Value, coord: []const *const mlir.Value, cache_policy: ?*const mlir.Value, mode: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.copy.SM107.tma_store_override", .{
        .operands = .{ .variadic = &.{
            &.{dst_desc},
            &.{src_smem_data_addr},
            &.{override_addr},
            tensor_size,
            lower_stride,
            if (upper_stride) |value| &.{value} else &.{},
            coord,
            if (cache_policy) |value| &.{value} else &.{},
        } },
        .attributes = &.{
            .named(ctx, "mode", mode),
        },
        .location = location,
    });
}

/// `cute_nvgpu.arch.copy.SM90.bulk_copy_g2s`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_copy_SM90_bulk_copy_g2s(ctx: *mlir.Context, gmem_data_addr: *const mlir.Value, dsmem_data_addr: *const mlir.Value, dsmem_bar_addr: *const mlir.Value, multicast_mask: ?*const mlir.Value, cache_policy: ?*const mlir.Value, size: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.copy.SM90.bulk_copy_g2s", .{
        .operands = .{ .variadic = &.{
            &.{gmem_data_addr},
            &.{dsmem_data_addr},
            &.{dsmem_bar_addr},
            if (multicast_mask) |value| &.{value} else &.{},
            if (cache_policy) |value| &.{value} else &.{},
        } },
        .attributes = &.{
            .named(ctx, "size", size),
        },
        .location = location,
    });
}

/// `cute_nvgpu.arch.copy.SM90.bulk_copy_s2g`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_copy_SM90_bulk_copy_s2g(ctx: *mlir.Context, smem_data_addr: *const mlir.Value, gmem_data_addr: *const mlir.Value, byte_mask: ?*const mlir.Value, cache_policy: ?*const mlir.Value, size: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.copy.SM90.bulk_copy_s2g", .{
        .operands = .{ .variadic = &.{
            &.{smem_data_addr},
            &.{gmem_data_addr},
            if (byte_mask) |value| &.{value} else &.{},
            if (cache_policy) |value| &.{value} else &.{},
        } },
        .attributes = &.{
            .named(ctx, "size", size),
        },
        .location = location,
    });
}

/// `cute_nvgpu.arch.copy.SM90.bulk_copy_s2s`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_copy_SM90_bulk_copy_s2s(ctx: *mlir.Context, smem_data_addr: *const mlir.Value, dsmem_data_addr: *const mlir.Value, dsmem_bar_addr: *const mlir.Value, cta_rank: *const mlir.Value, size: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.copy.SM90.bulk_copy_s2s", .{
        .operands = .{ .flat = &.{ smem_data_addr, dsmem_data_addr, dsmem_bar_addr, cta_rank } },
        .attributes = &.{
            .named(ctx, "size", size),
        },
        .location = location,
    });
}

/// `cute_nvgpu.arch.get_dyn_smem`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_get_dyn_smem(ctx: *mlir.Context, ptr_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.get_dyn_smem", .{
        .results = .{ .flat = &.{ptr_type} },
        .location = location,
    });
}

/// `cute_nvgpu.arch.get_dyn_smem_size`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_get_dyn_smem_size(ctx: *mlir.Context, size_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.get_dyn_smem_size", .{
        .results = .{ .flat = &.{size_type} },
        .location = location,
    });
}

/// `cute_nvgpu.arch.make_warp_uniform`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_make_warp_uniform(ctx: *mlir.Context, val: *const mlir.Value, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.make_warp_uniform", .{
        .operands = .{ .flat = &.{val} },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `cute_nvgpu.arch.mma.SM107.umma`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_mma_SM107_umma(ctx: *mlir.Context, a: *const mlir.Value, b: *const mlir.Value, cd: *const mlir.Value, idesc: *const mlir.Value, accum: *const mlir.Value, disable_output_lane: *const mlir.Value, num_cta: *const mlir.Attribute, scale: *const mlir.Attribute, a_type: *const mlir.Attribute, b_type: *const mlir.Attribute, shape_k: *const mlir.Attribute, a_collector_op: *const mlir.Attribute, b_collector_op: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.mma.SM107.umma", .{
        .operands = .{ .flat = &.{ a, b, cd, idesc, accum, disable_output_lane } },
        .attributes = &.{
            .named(ctx, "num_cta", num_cta),
            .named(ctx, "scale", scale),
            .named(ctx, "a_type", a_type),
            .named(ctx, "b_type", b_type),
            .named(ctx, "shape_k", shape_k),
            .named(ctx, "a_collector_op", a_collector_op),
            .named(ctx, "b_collector_op", b_collector_op),
        },
        .location = location,
    });
}

/// `cute_nvgpu.arch.mma.SM107.umma_block_scaled`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_mma_SM107_umma_block_scaled(ctx: *mlir.Context, a: *const mlir.Value, b: *const mlir.Value, cd: *const mlir.Value, sf_a: *const mlir.Value, sf_b: *const mlir.Value, idesc: *const mlir.Value, accum: *const mlir.Value, num_cta: *const mlir.Attribute, vec_size: *const mlir.Attribute, a_type: *const mlir.Attribute, b_type: *const mlir.Attribute, shape_k: *const mlir.Attribute, a_collector_op: *const mlir.Attribute, b_collector_op: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.mma.SM107.umma_block_scaled", .{
        .operands = .{ .flat = &.{ a, b, cd, sf_a, sf_b, idesc, accum } },
        .attributes = &.{
            .named(ctx, "num_cta", num_cta),
            .named(ctx, "vec_size", vec_size),
            .named(ctx, "a_type", a_type),
            .named(ctx, "b_type", b_type),
            .named(ctx, "shape_k", shape_k),
            .named(ctx, "a_collector_op", a_collector_op),
            .named(ctx, "b_collector_op", b_collector_op),
        },
        .location = location,
    });
}

/// `cute_nvgpu.arch.mma.SM107.umma_block_scaled_sparse`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_mma_SM107_umma_block_scaled_sparse(ctx: *mlir.Context, a: *const mlir.Value, b: *const mlir.Value, cd: *const mlir.Value, sf_a: *const mlir.Value, sf_b: *const mlir.Value, e_op: *const mlir.Value, idesc: *const mlir.Value, accum: *const mlir.Value, num_cta: *const mlir.Attribute, vec_size: *const mlir.Attribute, a_type: *const mlir.Attribute, b_type: *const mlir.Attribute, shape_k: *const mlir.Attribute, a_collector_op: *const mlir.Attribute, b_collector_op: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.mma.SM107.umma_block_scaled_sparse", .{
        .operands = .{ .flat = &.{ a, b, cd, sf_a, sf_b, e_op, idesc, accum } },
        .attributes = &.{
            .named(ctx, "num_cta", num_cta),
            .named(ctx, "vec_size", vec_size),
            .named(ctx, "a_type", a_type),
            .named(ctx, "b_type", b_type),
            .named(ctx, "shape_k", shape_k),
            .named(ctx, "a_collector_op", a_collector_op),
            .named(ctx, "b_collector_op", b_collector_op),
        },
        .location = location,
    });
}

/// `cute_nvgpu.arch.mma.SM107.umma_sparse`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_mma_SM107_umma_sparse(ctx: *mlir.Context, a: *const mlir.Value, b: *const mlir.Value, cd: *const mlir.Value, e: *const mlir.Value, idesc: *const mlir.Value, accum: *const mlir.Value, num_cta: *const mlir.Attribute, scale: *const mlir.Attribute, a_type: *const mlir.Attribute, b_type: *const mlir.Attribute, shape_k: *const mlir.Attribute, a_collector_op: *const mlir.Attribute, b_collector_op: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.mma.SM107.umma_sparse", .{
        .operands = .{ .flat = &.{ a, b, cd, e, idesc, accum } },
        .attributes = &.{
            .named(ctx, "num_cta", num_cta),
            .named(ctx, "scale", scale),
            .named(ctx, "a_type", a_type),
            .named(ctx, "b_type", b_type),
            .named(ctx, "shape_k", shape_k),
            .named(ctx, "a_collector_op", a_collector_op),
            .named(ctx, "b_collector_op", b_collector_op),
        },
        .location = location,
    });
}

/// `cute_nvgpu.arch.mma.SM120.block_scaled`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_mma_SM120_block_scaled(ctx: *mlir.Context, opA: []const *const mlir.Value, opB: []const *const mlir.Value, opC: []const *const mlir.Value, sfA: *const mlir.Value, sfB: *const mlir.Value, byte_id_a: ?*const mlir.Value, byte_id_b: ?*const mlir.Value, res_types: []const *const mlir.Type, shape_MNK: *const mlir.Attribute, vec_size: *const mlir.Attribute, thread_id_a: ?*const mlir.Attribute, thread_id_b: ?*const mlir.Attribute, a_type: *const mlir.Attribute, b_type: *const mlir.Attribute, sf_type: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 7) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "shape_MNK", shape_MNK));
    attributes.appendAssumeCapacity(.named(ctx, "vec_size", vec_size));
    if (thread_id_a) |value| attributes.appendAssumeCapacity(.named(ctx, "thread_id_a", value));
    if (thread_id_b) |value| attributes.appendAssumeCapacity(.named(ctx, "thread_id_b", value));
    attributes.appendAssumeCapacity(.named(ctx, "a_type", a_type));
    attributes.appendAssumeCapacity(.named(ctx, "b_type", b_type));
    attributes.appendAssumeCapacity(.named(ctx, "sf_type", sf_type));
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.mma.SM120.block_scaled", .{
        .operands = .{ .variadic = &.{
            opA,
            opB,
            opC,
            &.{sfA},
            &.{sfB},
            if (byte_id_a) |value| &.{value} else &.{},
            if (byte_id_b) |value| &.{value} else &.{},
        } },
        .results = .{ .variadic = &.{
            res_types,
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute_nvgpu.arch.prefetch_tma_desc`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_prefetch_tma_desc(ctx: *mlir.Context, tma_descriptor_ptr: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.prefetch_tma_desc", .{
        .operands = .{ .flat = &.{tma_descriptor_ptr} },
        .location = location,
    });
}

/// `cute_nvgpu.arch.sm100.alloc_tmem`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_sm100_alloc_tmem(ctx: *mlir.Context, n_cols: *const mlir.Value, dst_ptr: *const mlir.Value, is_two_cta: ?*const mlir.Attribute, exclusive: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (is_two_cta) |value| attributes.appendAssumeCapacity(.named(ctx, "is_two_cta", value));
    if (exclusive) |value| attributes.appendAssumeCapacity(.named(ctx, "exclusive", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.sm100.alloc_tmem", .{
        .operands = .{ .flat = &.{ n_cols, dst_ptr } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute_nvgpu.arch.sm100.dealloc_tmem`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_sm100_dealloc_tmem(ctx: *mlir.Context, tmem_ptr: *const mlir.Value, n_cols: *const mlir.Value, is_two_cta: ?*const mlir.Attribute, exclusive: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (is_two_cta) |value| attributes.appendAssumeCapacity(.named(ctx, "is_two_cta", value));
    if (exclusive) |value| attributes.appendAssumeCapacity(.named(ctx, "exclusive", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.sm100.dealloc_tmem", .{
        .operands = .{ .flat = &.{ tmem_ptr, n_cols } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute_nvgpu.arch.sm100.relinquish_tmem_alloc_permit`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_sm100_relinquish_tmem_alloc_permit(ctx: *mlir.Context, is_two_cta: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (is_two_cta) |value| attributes.appendAssumeCapacity(.named(ctx, "is_two_cta", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.sm100.relinquish_tmem_alloc_permit", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute_nvgpu.arch.sm100.retrieve_tmem_ptr`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_sm100_retrieve_tmem_ptr(ctx: *mlir.Context, smem_ptr: *const mlir.Value, tmem_ptr_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.sm100.retrieve_tmem_ptr", .{
        .operands = .{ .flat = &.{smem_ptr} },
        .results = .{ .flat = &.{tmem_ptr_type} },
        .location = location,
    });
}

/// `cute_nvgpu.atom.get_coord_tensor`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_get_coord_tensor(ctx: *mlir.Context, atom: *const mlir.Value, shape: *const mlir.Value, offset_b: ?*const mlir.Value, layout_b: ?*const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.get_coord_tensor", .{
        .operands = .{ .variadic = &.{
            &.{atom},
            &.{shape},
            if (offset_b) |value| &.{value} else &.{},
            if (layout_b) |value| &.{value} else &.{},
        } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute_nvgpu.atom.get_copy_s2t_smem_desc_view`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_get_copy_s2t_smem_desc_view(ctx: *mlir.Context, atom: *const mlir.Value, view: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.get_copy_s2t_smem_desc_view", .{
        .operands = .{ .flat = &.{ atom, view } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute_nvgpu.atom.get_value`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_get_value(ctx: *mlir.Context, atom: *const mlir.Value, result_type: *const mlir.Type, field: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.get_value", .{
        .operands = .{ .flat = &.{atom} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "field", field),
        },
        .location = location,
    });
}

/// `cute_nvgpu.atom.make_exec_tma`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_make_exec_tma(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, override: ?*const mlir.Attribute, no_fully_oob_tile: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (override) |value| attributes.appendAssumeCapacity(.named(ctx, "override", value));
    if (no_fully_oob_tile) |value| attributes.appendAssumeCapacity(.named(ctx, "no_fully_oob_tile", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.make_exec_tma", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute_nvgpu.atom.make_non_exec_2d_gather4_tma_load`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_make_non_exec_2d_gather4_tma_load(ctx: *mlir.Context, gmem_tensor: *const mlir.Value, gmem_coord_layout: *const mlir.Value, smem_layout: *const mlir.Value, cta_v_map: *const mlir.Value, non_exec_atom_type: *const mlir.Type, tma_tensor_type: *const mlir.Type, kind: *const mlir.Attribute, num_multicast: ?*const mlir.Attribute, tma_format: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (num_multicast) |value| attributes.appendAssumeCapacity(.named(ctx, "num_multicast", value));
    if (tma_format) |value| attributes.appendAssumeCapacity(.named(ctx, "tma_format", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.make_non_exec_2d_gather4_tma_load", .{
        .operands = .{ .flat = &.{ gmem_tensor, gmem_coord_layout, smem_layout, cta_v_map } },
        .results = .{ .flat = &.{ non_exec_atom_type, tma_tensor_type } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute_nvgpu.atom.make_non_exec_2d_scatter4_tma_store`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_make_non_exec_2d_scatter4_tma_store(ctx: *mlir.Context, gmem_tensor: *const mlir.Value, gmem_index_layout: *const mlir.Value, smem_layout: *const mlir.Value, cta_v_map: *const mlir.Value, non_exec_atom_type: *const mlir.Type, tma_tensor_type: *const mlir.Type, tma_format: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (tma_format) |value| attributes.appendAssumeCapacity(.named(ctx, "tma_format", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.make_non_exec_2d_scatter4_tma_store", .{
        .operands = .{ .flat = &.{ gmem_tensor, gmem_index_layout, smem_layout, cta_v_map } },
        .results = .{ .flat = &.{ non_exec_atom_type, tma_tensor_type } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute_nvgpu.atom.make_non_exec_im2col_tma_load`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_make_non_exec_im2col_tma_load(ctx: *mlir.Context, gmem_tensor: *const mlir.Value, smem_layout: *const mlir.Value, cta_v_map: *const mlir.Value, lower_corner_whd: *const mlir.Value, upper_corner_whd: *const mlir.Value, lower_padding_whd: *const mlir.Value, upper_padding_whd: *const mlir.Value, stride_whd: *const mlir.Value, lower_srt: *const mlir.Value, stride_srt: *const mlir.Value, non_exec_atom_type: *const mlir.Type, tma_tensor_type: *const mlir.Type, kind: *const mlir.Attribute, num_multicast: ?*const mlir.Attribute, tma_format: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (num_multicast) |value| attributes.appendAssumeCapacity(.named(ctx, "num_multicast", value));
    if (tma_format) |value| attributes.appendAssumeCapacity(.named(ctx, "tma_format", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.make_non_exec_im2col_tma_load", .{
        .operands = .{ .flat = &.{ gmem_tensor, smem_layout, cta_v_map, lower_corner_whd, upper_corner_whd, lower_padding_whd, upper_padding_whd, stride_whd, lower_srt, stride_srt } },
        .results = .{ .flat = &.{ non_exec_atom_type, tma_tensor_type } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute_nvgpu.atom.make_non_exec_im2col_tma_store`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_make_non_exec_im2col_tma_store(ctx: *mlir.Context, gmem_tensor: *const mlir.Value, smem_layout: *const mlir.Value, cta_v_map: *const mlir.Value, non_exec_atom_type: *const mlir.Type, tma_tensor_type: *const mlir.Type, tma_format: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (tma_format) |value| attributes.appendAssumeCapacity(.named(ctx, "tma_format", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.make_non_exec_im2col_tma_store", .{
        .operands = .{ .flat = &.{ gmem_tensor, smem_layout, cta_v_map } },
        .results = .{ .flat = &.{ non_exec_atom_type, tma_tensor_type } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute_nvgpu.atom.make_non_exec_tiled_tma_load`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_make_non_exec_tiled_tma_load(ctx: *mlir.Context, gmem_tensor: *const mlir.Value, smem_layout: *const mlir.Value, cta_v_map: *const mlir.Value, non_exec_atom_type: *const mlir.Type, tma_tensor_type: *const mlir.Type, kind: *const mlir.Attribute, num_multicast: ?*const mlir.Attribute, tma_format: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (num_multicast) |value| attributes.appendAssumeCapacity(.named(ctx, "num_multicast", value));
    if (tma_format) |value| attributes.appendAssumeCapacity(.named(ctx, "tma_format", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.make_non_exec_tiled_tma_load", .{
        .operands = .{ .flat = &.{ gmem_tensor, smem_layout, cta_v_map } },
        .results = .{ .flat = &.{ non_exec_atom_type, tma_tensor_type } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute_nvgpu.atom.make_non_exec_tiled_tma_reduce`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_make_non_exec_tiled_tma_reduce(ctx: *mlir.Context, gmem_tensor: *const mlir.Value, smem_layout: *const mlir.Value, cta_v_map: *const mlir.Value, non_exec_atom_type: *const mlir.Type, tma_tensor_type: *const mlir.Type, kind: *const mlir.Attribute, tma_format: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (tma_format) |value| attributes.appendAssumeCapacity(.named(ctx, "tma_format", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.make_non_exec_tiled_tma_reduce", .{
        .operands = .{ .flat = &.{ gmem_tensor, smem_layout, cta_v_map } },
        .results = .{ .flat = &.{ non_exec_atom_type, tma_tensor_type } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute_nvgpu.atom.make_non_exec_tiled_tma_store`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_make_non_exec_tiled_tma_store(ctx: *mlir.Context, gmem_tensor: *const mlir.Value, smem_layout: *const mlir.Value, cta_v_map: *const mlir.Value, non_exec_atom_type: *const mlir.Type, tma_tensor_type: *const mlir.Type, tma_format: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (tma_format) |value| attributes.appendAssumeCapacity(.named(ctx, "tma_format", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.make_non_exec_tiled_tma_store", .{
        .operands = .{ .flat = &.{ gmem_tensor, smem_layout, cta_v_map } },
        .results = .{ .flat = &.{ non_exec_atom_type, tma_tensor_type } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute_nvgpu.atom.make_s2t_copy`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_make_s2t_copy(ctx: *mlir.Context, copy_s2t_atom: *const mlir.Value, tmem_memref: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.make_s2t_copy", .{
        .operands = .{ .flat = &.{ copy_s2t_atom, tmem_memref } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute_nvgpu.atom.make_tma_residue_tensor`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_make_tma_residue_tensor(ctx: *mlir.Context, tma_atom: *const mlir.Value, gmem_tensor: *const mlir.Value, residue_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.make_tma_residue_tensor", .{
        .operands = .{ .flat = &.{ tma_atom, gmem_tensor } },
        .results = .{ .flat = &.{residue_type} },
        .location = location,
    });
}

/// `cute_nvgpu.atom.make_tmem_copy`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_make_tmem_copy(ctx: *mlir.Context, atom: *const mlir.Value, tmem_memref: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.make_tmem_copy", .{
        .operands = .{ .flat = &.{ atom, tmem_memref } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute_nvgpu.atom.set_value`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_set_value(ctx: *mlir.Context, atom: *const mlir.Value, value_: *const mlir.Value, result_type: *const mlir.Type, field: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.set_value", .{
        .operands = .{ .flat = &.{ atom, value_ } },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "field", field),
        },
        .location = location,
    });
}

/// `cute_nvgpu.atom.sm107.get_copy_s2t_smem_desc_view`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_sm107_get_copy_s2t_smem_desc_view(ctx: *mlir.Context, atom: *const mlir.Value, view: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.sm107.get_copy_s2t_smem_desc_view", .{
        .operands = .{ .flat = &.{ atom, view } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute_nvgpu.atom.tma_partition`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_tma_partition(ctx: *mlir.Context, tma_atom: *const mlir.Value, cta_coord: *const mlir.Value, cta_layout: *const mlir.Value, smem_tensor: *const mlir.Value, target_tensors: []const *const mlir.Value, res_smem_tensor_type: *const mlir.Type, res_target_tensors_types: []const *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.tma_partition", .{
        .operands = .{ .variadic = &.{
            &.{tma_atom},
            &.{cta_coord},
            &.{cta_layout},
            &.{smem_tensor},
            target_tensors,
        } },
        .results = .{ .variadic = &.{
            &.{res_smem_tensor_type},
            res_target_tensors_types,
        } },
        .location = location,
    });
}

/// `cute_nvgpu.cast_tma_desc_to_integer`. Result types are explicit; attributes use mlir.Attribute.
pub fn cast_tma_desc_to_integer(ctx: *mlir.Context, src: *const mlir.Value, dst_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.cast_tma_desc_to_integer", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .location = location,
    });
}

/// `cute_nvgpu.copy_tma_desc`. Result types are explicit; attributes use mlir.Attribute.
pub fn copy_tma_desc(ctx: *mlir.Context, atom: *const mlir.Value, tma_desc_ptr: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.copy_tma_desc", .{
        .operands = .{ .flat = &.{ atom, tma_desc_ptr } },
        .location = location,
    });
}

/// `cute_nvgpu.get_grid_constant_pointer`. Result types are explicit; attributes use mlir.Attribute.
pub fn get_grid_constant_pointer(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.get_grid_constant_pointer", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute_nvgpu.get_tma_desc_addr`. Result types are explicit; attributes use mlir.Attribute.
pub fn get_tma_desc_addr(ctx: *mlir.Context, src: *const mlir.Value, ptr_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.get_tma_desc_addr", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{ptr_type} },
        .location = location,
    });
}

/// `cute_nvgpu.make_gmma_smem_desc`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_gmma_smem_desc(ctx: *mlir.Context, src: *const mlir.Value, res_type: *const mlir.Type, layout: *const mlir.Attribute, major: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.make_gmma_smem_desc", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{res_type} },
        .attributes = &.{
            .named(ctx, "layout", layout),
            .named(ctx, "major", major),
        },
        .location = location,
    });
}

/// `cute_nvgpu.make_tma_desc_im2col`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_tma_desc_im2col(ctx: *mlir.Context, gmem_view: *const mlir.Value, smem_layout: *const mlir.Value, cta_value_tile: *const mlir.Value, lower_corner_whd: *const mlir.Value, upper_corner_whd: *const mlir.Value, lower_padding_whd: *const mlir.Value, upper_padding_whd: *const mlir.Value, stride_whd: *const mlir.Value, lower_srt: *const mlir.Value, stride_srt: *const mlir.Value, result_type: *const mlir.Type, num_multicast: ?*const mlir.Attribute, tma_format: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (num_multicast) |value| attributes.appendAssumeCapacity(.named(ctx, "num_multicast", value));
    if (tma_format) |value| attributes.appendAssumeCapacity(.named(ctx, "tma_format", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.make_tma_desc_im2col", .{
        .operands = .{ .flat = &.{ gmem_view, smem_layout, cta_value_tile, lower_corner_whd, upper_corner_whd, lower_padding_whd, upper_padding_whd, stride_whd, lower_srt, stride_srt } },
        .results = .{ .flat = &.{result_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute_nvgpu.make_tma_desc_im2col_at`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_tma_desc_im2col_at(ctx: *mlir.Context, gmem_view: *const mlir.Value, smem_layout: *const mlir.Value, cta_value_tile: *const mlir.Value, lower_corner_whd: *const mlir.Value, upper_corner_whd: *const mlir.Value, lower_padding_whd: *const mlir.Value, upper_padding_whd: *const mlir.Value, stride_whd: *const mlir.Value, lower_srt: *const mlir.Value, stride_srt: *const mlir.Value, tma_desc_addr: *const mlir.Value, num_multicast: ?*const mlir.Attribute, tma_format: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (num_multicast) |value| attributes.appendAssumeCapacity(.named(ctx, "num_multicast", value));
    if (tma_format) |value| attributes.appendAssumeCapacity(.named(ctx, "tma_format", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.make_tma_desc_im2col_at", .{
        .operands = .{ .flat = &.{ gmem_view, smem_layout, cta_value_tile, lower_corner_whd, upper_corner_whd, lower_padding_whd, upper_padding_whd, stride_whd, lower_srt, stride_srt, tma_desc_addr } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute_nvgpu.make_tma_desc_tiled`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_tma_desc_tiled(ctx: *mlir.Context, gmem_view: *const mlir.Value, smem_layout: *const mlir.Value, cta_value_tile: *const mlir.Value, traversal_stride: ?*const mlir.Value, result_type: *const mlir.Type, mode: *const mlir.Attribute, num_multicast: ?*const mlir.Attribute, tma_format: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "mode", mode));
    if (num_multicast) |value| attributes.appendAssumeCapacity(.named(ctx, "num_multicast", value));
    if (tma_format) |value| attributes.appendAssumeCapacity(.named(ctx, "tma_format", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.make_tma_desc_tiled", .{
        .operands = .{ .variadic = &.{
            &.{gmem_view},
            &.{smem_layout},
            &.{cta_value_tile},
            if (traversal_stride) |value| &.{value} else &.{},
        } },
        .results = .{ .flat = &.{result_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute_nvgpu.make_tma_desc_tiled_at`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_tma_desc_tiled_at(ctx: *mlir.Context, gmem_view: *const mlir.Value, smem_layout: *const mlir.Value, cta_value_tile: *const mlir.Value, tma_desc_addr: *const mlir.Value, traversal_stride: ?*const mlir.Value, mode: *const mlir.Attribute, num_multicast: ?*const mlir.Attribute, tma_format: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "mode", mode));
    if (num_multicast) |value| attributes.appendAssumeCapacity(.named(ctx, "num_multicast", value));
    if (tma_format) |value| attributes.appendAssumeCapacity(.named(ctx, "tma_format", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.make_tma_desc_tiled_at", .{
        .operands = .{ .variadic = &.{
            &.{gmem_view},
            &.{smem_layout},
            &.{cta_value_tile},
            &.{tma_desc_addr},
            if (traversal_stride) |value| &.{value} else &.{},
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute_nvgpu.make_umma_smem_desc`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_umma_smem_desc(ctx: *mlir.Context, src: *const mlir.Value, nextSrc: ?*const mlir.Value, res_type: *const mlir.Type, layout: *const mlir.Attribute, major: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.make_umma_smem_desc", .{
        .operands = .{ .variadic = &.{
            &.{src},
            if (nextSrc) |value| &.{value} else &.{},
        } },
        .results = .{ .flat = &.{res_type} },
        .attributes = &.{
            .named(ctx, "layout", layout),
            .named(ctx, "major", major),
        },
        .location = location,
    });
}

/// `cute_nvgpu.prefetch_tma_desc`. Result types are explicit; attributes use mlir.Attribute.
pub fn prefetch_tma_desc(ctx: *mlir.Context, atom: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.prefetch_tma_desc", .{
        .operands = .{ .flat = &.{atom} },
        .location = location,
    });
}

/// `cute_nvgpu.sm103.make_umma_smem_desc_circular`. Result types are explicit; attributes use mlir.Attribute.
pub fn sm103_make_umma_smem_desc_circular(ctx: *mlir.Context, src: *const mlir.Value, nextSrc: ?*const mlir.Value, res_type: *const mlir.Type, layout: *const mlir.Attribute, major: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.sm103.make_umma_smem_desc_circular", .{
        .operands = .{ .variadic = &.{
            &.{src},
            if (nextSrc) |value| &.{value} else &.{},
        } },
        .results = .{ .flat = &.{res_type} },
        .attributes = &.{
            .named(ctx, "layout", layout),
            .named(ctx, "major", major),
        },
        .location = location,
    });
}

/// `cute_nvgpu.sm107.make_umma_smem_desc`. Result types are explicit; attributes use mlir.Attribute.
pub fn sm107_make_umma_smem_desc(ctx: *mlir.Context, src: *const mlir.Value, nextSrc: ?*const mlir.Value, res_type: *const mlir.Type, layout: *const mlir.Attribute, major: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.sm107.make_umma_smem_desc", .{
        .operands = .{ .variadic = &.{
            &.{src},
            if (nextSrc) |value| &.{value} else &.{},
        } },
        .results = .{ .flat = &.{res_type} },
        .attributes = &.{
            .named(ctx, "layout", layout),
            .named(ctx, "major", major),
        },
        .location = location,
    });
}

/// `cute_nvgpu.sm107.make_umma_smem_desc_circular`. Result types are explicit; attributes use mlir.Attribute.
pub fn sm107_make_umma_smem_desc_circular(ctx: *mlir.Context, src: *const mlir.Value, nextSrc: ?*const mlir.Value, res_type: *const mlir.Type, layout: *const mlir.Attribute, major: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.sm107.make_umma_smem_desc_circular", .{
        .operands = .{ .variadic = &.{
            &.{src},
            if (nextSrc) |value| &.{value} else &.{},
        } },
        .results = .{ .flat = &.{res_type} },
        .attributes = &.{
            .named(ctx, "layout", layout),
            .named(ctx, "major", major),
        },
        .location = location,
    });
}

/// `cute_nvgpu.update_tma_desc`. Result types are explicit; attributes use mlir.Attribute.
pub fn update_tma_desc(ctx: *mlir.Context, tma_atom: *const mlir.Value, gmem_tensor: *const mlir.Value, tma_desc_ptr: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.update_tma_desc", .{
        .operands = .{ .flat = &.{ tma_atom, gmem_tensor, tma_desc_ptr } },
        .location = location,
    });
}

pub const operation_names: []const []const u8 = &.{
    "cute_nvgpu.arch.alloc_rmem",
    "cute_nvgpu.arch.alloc_smem",
    "cute_nvgpu.arch.copy.SM100.copy_s2t",
    "cute_nvgpu.arch.copy.SM100.tma_load",
    "cute_nvgpu.arch.copy.SM100.tma_reduce",
    "cute_nvgpu.arch.copy.SM100.tma_store",
    "cute_nvgpu.arch.copy.SM100.tmem_load",
    "cute_nvgpu.arch.copy.SM100.tmem_store",
    "cute_nvgpu.arch.copy.SM107.tma_load_override",
    "cute_nvgpu.arch.copy.SM107.tma_store_override",
    "cute_nvgpu.arch.copy.SM90.bulk_copy_g2s",
    "cute_nvgpu.arch.copy.SM90.bulk_copy_s2g",
    "cute_nvgpu.arch.copy.SM90.bulk_copy_s2s",
    "cute_nvgpu.arch.get_dyn_smem",
    "cute_nvgpu.arch.get_dyn_smem_size",
    "cute_nvgpu.arch.make_warp_uniform",
    "cute_nvgpu.arch.mma.SM107.umma",
    "cute_nvgpu.arch.mma.SM107.umma_block_scaled",
    "cute_nvgpu.arch.mma.SM107.umma_block_scaled_sparse",
    "cute_nvgpu.arch.mma.SM107.umma_sparse",
    "cute_nvgpu.arch.mma.SM120.block_scaled",
    "cute_nvgpu.arch.prefetch_tma_desc",
    "cute_nvgpu.arch.sm100.alloc_tmem",
    "cute_nvgpu.arch.sm100.dealloc_tmem",
    "cute_nvgpu.arch.sm100.relinquish_tmem_alloc_permit",
    "cute_nvgpu.arch.sm100.retrieve_tmem_ptr",
    "cute_nvgpu.atom.get_coord_tensor",
    "cute_nvgpu.atom.get_copy_s2t_smem_desc_view",
    "cute_nvgpu.atom.get_value",
    "cute_nvgpu.atom.make_exec_tma",
    "cute_nvgpu.atom.make_non_exec_2d_gather4_tma_load",
    "cute_nvgpu.atom.make_non_exec_2d_scatter4_tma_store",
    "cute_nvgpu.atom.make_non_exec_im2col_tma_load",
    "cute_nvgpu.atom.make_non_exec_im2col_tma_store",
    "cute_nvgpu.atom.make_non_exec_tiled_tma_load",
    "cute_nvgpu.atom.make_non_exec_tiled_tma_reduce",
    "cute_nvgpu.atom.make_non_exec_tiled_tma_store",
    "cute_nvgpu.atom.make_s2t_copy",
    "cute_nvgpu.atom.make_tma_residue_tensor",
    "cute_nvgpu.atom.make_tmem_copy",
    "cute_nvgpu.atom.set_value",
    "cute_nvgpu.atom.sm107.get_copy_s2t_smem_desc_view",
    "cute_nvgpu.atom.tma_partition",
    "cute_nvgpu.cast_tma_desc_to_integer",
    "cute_nvgpu.copy_tma_desc",
    "cute_nvgpu.get_grid_constant_pointer",
    "cute_nvgpu.get_tma_desc_addr",
    "cute_nvgpu.make_gmma_smem_desc",
    "cute_nvgpu.make_tma_desc_im2col",
    "cute_nvgpu.make_tma_desc_im2col_at",
    "cute_nvgpu.make_tma_desc_tiled",
    "cute_nvgpu.make_tma_desc_tiled_at",
    "cute_nvgpu.make_umma_smem_desc",
    "cute_nvgpu.prefetch_tma_desc",
    "cute_nvgpu.sm103.make_umma_smem_desc_circular",
    "cute_nvgpu.sm107.make_umma_smem_desc",
    "cute_nvgpu.sm107.make_umma_smem_desc_circular",
    "cute_nvgpu.update_tma_desc",
};
