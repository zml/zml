//! Zig bindings of NVIDIA's NVVM dialect of the CuTe DSL (nvidia-cutlass-dsl 4.8.0), generated
//! from the dialect's .td files; do not edit.

const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");
const stdx = @import("stdx");

pub const dialect_namespace = "nvvm";
/// The C dialect handle, `mlirGetDialectHandle__cute_nvvm__`.
pub const dialect_handle = "cute_nvvm";

// Enums, generated from the dialect's .td files; values are the compiler's.

/// operations supported by atom instruction.
pub const AtomicOpKind = enum(u32) {
    @"and" = 0,
    @"or" = 1,
    xor = 2,
    cas = 3,
    exch = 4,
    add = 5,
    inc = 6,
    dec = 7,
    min = 8,
    max = 9,
    umin = 10,
    umax = 11,
    fadd = 12,
    sub = 13,
};

/// NVVM barrier reduction operation.
pub const BarrierReduction = enum(u32) {
    popc = 0,
    @"and" = 1,
    @"or" = 2,
};

/// NVVM barrier redux kind.
pub const BarrierReduxKind = enum(u32) {
    @"and" = 0,
    @"or" = 1,
    popc = 2,
};

/// MMA Block Scale Format.
pub const BlockScaleFormat = enum(u32) {
    ue8m0 = 0,
    ue4m3 = 1,
};

/// NVVM CTA group kind.
pub const CTAGroupKind = enum(u32) {
    cta_1 = 0,
    cta_2 = 1,
};

/// NVVM CVT Pack Float kind.
pub const CVTPackFloatKind = enum(u32) {
    f32 = 0,
    f16x2 = 1,
    e4m3x2 = 2,
    e5m2x2 = 3,
    bf16x2 = 4,
    e2m1x2 = 5,
    e2m3x2 = 6,
    e3m2x2 = 7,
    ue8m0x2 = 8,
};

/// NVVM Cache Eviction Priority.
pub const CacheEvictionPriority = enum(u32) {
    evict_normal = 0,
    evict_first = 1,
    evict_last = 2,
    evict_unchanged = 3,
    no_allocate = 4,
};

/// NVVM ClusterLaunchControlQueryType.
pub const ClusterLaunchControlQueryType = enum(u32) {
    is_canceled = 0,
    get_first_cta_id_x = 1,
    get_first_cta_id_y = 2,
    get_first_cta_id_z = 3,
};

/// Comparison operator encoding.
pub const CompareOpKind = enum(u32) {
    eq = 0,
    ne = 1,
    ugt = 2,
    uge = 3,
    ult = 4,
    ule = 5,
    sgt = 6,
    sge = 7,
    slt = 8,
    sle = 9,
};

/// NVVM ConvertFP4Type kind.
pub const ConvertFP4Type = enum(u32) {
    e2m1 = 0,
};

/// NVVM ConvertFP8Type kind.
pub const ConvertFP8Type = enum(u32) {
    e4m3 = 0,
    e5m2 = 1,
    ue8m0 = 2,
    ue5m3 = 3,
};

/// NVVM ConvertScale kind.
pub const ConvertScaleKind = enum(u32) {
    none = 0,
    ue8m0 = 1,
};

/// NVVM DotAccumulateType.
pub const DotAccumulateType = enum(u32) {
    signed = 1,
    unsigned = 0,
};

/// NVVM L2 Prefetch Size.
pub const EvictKind = enum(u32) {
    evict_normal = 0,
    evict_first = 1,
    evict_last = 2,
    evict_unchanged = 3,
    evict_no_allocate = 4,
    none = 5,
};

/// NVVM FPRoundingMode kind.
pub const FPRoundingMode = enum(u32) {
    none = 0,
    rn = 1,
    rm = 2,
    rp = 3,
    rz = 4,
    rna = 5,
    rs = 6,
};

/// Action kind for grid dependency control.
pub const GridDepActionKind = enum(u32) {
    wait = 0,
    launch_dependents = 1,
};

/// NVVM IntegerRoundingMode kind.
pub const IntegerRoundingMode = enum(u32) {
    none = 0,
    rni = 1,
    rzi = 2,
    rmi = 3,
    rpi = 4,
};

/// NVVM L2 Prefetch Size.
pub const L2PrefetchSize = enum(u32) {
    none = 0,
    reserved = 1,
    size_64b = 2,
    size_128b = 3,
    size_256b = 4,
};

/// Element type for ldmatrix and stmatrix.
pub const LdStMatrixEltType = enum(u32) {
    b16 = 0,
    b8 = 1,
    @"b8x16.b6x16_p32" = 2,
    @"b8x16.b4x16_p64" = 3,
};

/// NVVM load cache modifier kind(Ext).
pub const LoadCacheModifierExtKind = enum(u32) {
    ca = 0,
    cg = 1,
    cs = 2,
    lu = 3,
    cv = 4,
    none = 5,
};

/// NVVM load cache modifier kind.
pub const LoadCacheModifierKind = enum(u32) {
    ca = 0,
    cg = 1,
    cs = 2,
    lu = 3,
    cv = 4,
};

/// shape attribute for ldmatrix.
pub const LoadShape = enum(u32) {
    m8n8 = 0,
    m8n16 = 1,
    m16n16 = 3,
};

/// source format for ldmatrix.
pub const LoadSrcFormat = enum(u32) {
    b6x16_p32 = 0,
    b4x16_p64 = 1,
    b8 = 2,
};

/// NVVM MBarrier Layout.
pub const MBarrierLayout = enum(u32) {
    v0 = 0,
    v1 = 1,
};

/// NVVM mbarrier phase type.
pub const MBarrierPhase = enum(u32) {
    none = 0,
    primary = 1,
    conditional = 2,
};

/// NVVM MBarrier scope kind.
pub const MBarrierScopeKind = enum(u32) {
    cta = 0,
    cluster = 1,
};

/// NVVM MBarrier space kind.
pub const MBarrierSpaceKind = enum(u32) {
    cta = 0,
    cluster = 1,
};

/// NVVM MBarrier Transaction kind.
pub const MBarrierTxnKind = enum(u32) {
    arrive = 0,
    arrive_drop = 1,
    arrive_expect_tx = 2,
    arrive_drop_expect_tx = 3,
    expect_tx = 4,
    complete_tx = 5,
};

/// NVVM MBarrier wait kind.
pub const MBarrierWaitKind = enum(u32) {
    @"test" = 0,
    @"try" = 1,
};

/// MMA binary operations.
pub const MMAB1Op = enum(u32) {
    none = 0,
    xor_popc = 1,
    and_popc = 2,
};

/// Block Scale Kind.
pub const MMABlockScaleKind = enum(u32) {
    mxf8f6f4 = 0,
    mxf4 = 1,
    mxf4nvf4 = 2,
};

/// MMA CTA count.
pub const MMACtaCount = enum(u32) {
    cta1 = 1,
    cta2 = 2,
};

/// NVVM MMA frag type.
pub const MMAFrag = enum(u32) {
    a = 0,
    b = 1,
    c = 2,
};

/// MMA overflow options.
pub const MMAIntOverflow = enum(u32) {
    satfinite = 1,
    wrapped = 0,
};

/// MMA operation kind.
pub const MMAKind = enum(u32) {
    f8f6f4 = 0,
};

/// NVVM MMA layout.
pub const MMALayout = enum(u32) {
    row = 0,
    col = 1,
};

/// NVVM MMA types.
pub const MMATypes = enum(u32) {
    f16 = 0,
    f32 = 1,
    tf32 = 2,
    bf16 = 9,
    s8 = 4,
    u8 = 3,
    s32 = 5,
    s4 = 8,
    u4 = 7,
    b1 = 6,
    f64 = 10,
    e4m3 = 11,
    e5m2 = 12,
    e3m2 = 13,
    e2m3 = 14,
    e2m1 = 15,
};

/// NVVM match sync kind.
pub const MatchSyncKind = enum(u32) {
    any = 0,
    all = 1,
};

/// NVVM Memory Ordering kind.
pub const MemOrderKind = enum(u32) {
    weak = 0,
    relaxed = 1,
    acquire = 2,
    release = 3,
    acq_rel = 4,
    sc = 5,
    mmio = 6,
    @"volatile" = 8,
    constant = 7,
};

/// NVVM Memory Scope kind.
pub const MemScopeKind = enum(u32) {
    cta = 0,
    cluster = 1,
    gpu = 2,
    sys = 3,
};

/// multiply mode attribute.
pub const MulMode = enum(u32) {
    hi = 0,
    lo = 1,
    wide = 2,
};

/// NVVM Memory Space.
pub const NVVMMemorySpace = enum(u32) {
    generic = 0,
    global = 1,
    shared = 3,
    constant = 4,
    local = 5,
    tensor = 6,
    shared_cluster = 7,
};

/// NVVM permute mode.
pub const PermuteMode = enum(u32) {
    default = 0,
    f4e = 1,
    b4e = 2,
    rc8 = 3,
    ecl = 4,
    ecr = 5,
    rc16 = 6,
};

/// NVVM Prefetch Cache Level.
pub const PrefetchCacheLevel = enum(u32) {
    L1 = 0,
    L2 = 1,
};

/// Proxy kind.
pub const ProxyKind = enum(u32) {
    alias = 0,
    async = 1,
    @"async.global" = 2,
    @"async.shared" = 3,
    tensormap = 4,
    generic = 5,
};

/// NVVM Reduction Kind attribute.
pub const ReductionKind = enum(u32) {
    add = 1,
    @"and" = 2,
    max = 3,
    min = 4,
    @"or" = 5,
    umax = 6,
    umin = 7,
    xor = 8,
    fmin = 9,
    fmax = 10,
};

/// Ops supported by red instruction.
pub const ReductionOp = enum(u32) {
    @"and" = 0,
    @"or" = 1,
    xor = 2,
    add = 3,
    inc = 4,
    dec = 5,
    min = 6,
    max = 7,
};

/// types supported by red instruction.
pub const ReductionType = enum(u32) {
    b32 = 0,
    b64 = 1,
    u32 = 2,
    u64 = 3,
    s32 = 4,
    s64 = 5,
    f32 = 6,
    f64 = 7,
    f16 = 8,
    f16x2 = 9,
    bf16 = 10,
    bf16x2 = 11,
};

/// Sparse tensor compression element size.
pub const SPCompressElemSize = enum(u32) {
    b4 = 4,
    b8 = 8,
    b16 = 16,
};

/// Sparse tensor compression factor type.
pub const SPCompressFactorType = enum(u32) {
    sp2to4 = 0,
};

/// Sparse tensor compression index size.
pub const SPCompressIndexSize = enum(u32) {
    b2 = 2,
    b4 = 4,
};

/// spcompress operation kind.
pub const SPCompressOpKind = enum(u32) {
    min = 0,
    max = 1,
};

/// Sparse tensor compression repetition factor.
pub const SPCompressRepFactor = enum(u32) {
    x1 = 1,
    x2 = 2,
    x4 = 4,
    x8 = 8,
    x16 = 16,
    x32 = 32,
    x64 = 64,
    x128 = 128,
};

/// Sparse tensor decompression element size.
pub const SPDecompressElemSize = enum(u32) {
    b4 = 4,
    b8 = 8,
    b16 = 16,
};

/// Sparse tensor decompression factor type.
pub const SPDecompressFactorType = enum(u32) {
    sp1to2 = 0,
    sp1to4 = 1,
    sp1to8 = 2,
    sp1to16 = 3,
    sp2to4 = 4,
    sp2to8 = 5,
    sp2to16 = 6,
    sp4to8 = 7,
    sp4to16 = 8,
    sp8to16 = 9,
};

/// Sparse tensor decompression index size.
pub const SPDecompressIndexSize = enum(u32) {
    b2 = 2,
    b4 = 4,
};

/// Sparse tensor decompression repetition factor.
pub const SPDecompressRepFactor = enum(u32) {
    x1 = 1,
    x2 = 2,
    x4 = 4,
    x8 = 8,
    x16 = 16,
    x32 = 32,
    x64 = 64,
};

/// NVVM SaturationMode kind.
pub const SaturationMode = enum(u32) {
    none = 0,
    satfinite = 1,
    sat = 2,
};

/// NVVM SaturationMode kind.
pub const SaturationModeKind = enum(u32) {
    none = 0,
    satfinite = 1,
};

/// MMA Scale Vector Sizes.
pub const ScaleVecSize = enum(u32) {
    x1 = 0,
    x2 = 1,
    x4 = 2,
};

/// NVVM set max register action.
pub const SetMaxRegisterAction = enum(u32) {
    decrease = 1,
    increase = 0,
};

/// Shared memory space.
pub const SharedSpace = enum(u32) {
    cta = 0,
    cluster = 1,
};

/// NVVM shuffle kind.
pub const ShflKind = enum(u32) {
    bfly = 0,
    up = 1,
    down = 2,
    idx = 3,
};

/// MMA Sparsity Format.
pub const SparsityFormat = enum(u32) {
    thread = 0,
};

/// NVVM State Space.
pub const StateSpace = enum(u32) {
    generic = 0,
    global = 1,
    shared_cta = 2,
    shared_cluster = 3,
    constant = 4,
    local = 5,
    tensor = 6,
};

/// NVVM store cache modifier kind.
pub const StoreCacheModifierKind = enum(u32) {
    wb = 0,
    cg = 1,
    cs = 2,
    wt = 3,
    none = 4,
};

/// shape attribute for stmatrix.
pub const StoreShape = enum(u32) {
    m8n8 = 0,
    m16n8 = 2,
};

/// Cluster MMA Barrier Parameter Type.
pub const TCBarParam = enum(u32) {
    a1t0 = 0,
    a0tx = 1,
};

/// NVVM TMA Load Mode.
pub const TMALoadMode = enum(u32) {
    tile = 0,
    im2col = 1,
    im2col_w = 2,
    im2col_w_128 = 3,
    tile_gather4 = 4,
};

/// NVVM TMA redux kind.
pub const TMAReduxKind = enum(u32) {
    add = 0,
    max = 2,
    min = 1,
    inc = 3,
    dec = 4,
    @"and" = 5,
    @"or" = 6,
    xor = 7,
};

/// NVVM TMA Store Mode.
pub const TMAStoreMode = enum(u32) {
    tile = 0,
    im2col = 1,
    tile_scatter4 = 2,
    im2col_w = 3,
};

/// tcgen05 cp multicast.
pub const Tcgen05CpMulticast = enum(u32) {
    none = 0,
    warpx2_02_13 = 1,
    warpx2_01_23 = 2,
    warpx4 = 3,
};

/// tcgen05 cp shapes.
pub const Tcgen05CpShape = enum(u32) {
    shape_128x256b = 0,
    shape_4x256b = 1,
    shape_128x128b = 2,
    shape_64x128b = 3,
    shape_32x128b = 4,
};

/// tcgen05 cp source format.
pub const Tcgen05CpSrcFormat = enum(u32) {
    b6x16_p32 = 0,
    b4x16_p64 = 1,
};

/// NVVM Tcgen05 fence kind.
pub const Tcgen05FenceKind = enum(u32) {
    before = 0,
    after = 1,
};

/// allowed 32-bit signless integer cases: 0, 1, 2, 3, 4.
pub const Tcgen05LdStShape = enum(u32) {
    shape_16x64b = 0,
    shape_16x128b = 1,
    shape_16x256b = 2,
    shape_32x32b = 3,
    shape_16x32bx2 = 4,
};

/// tcgen05.mma block scale attribute.
pub const Tcgen05MMABlockScale = enum(u32) {
    default = 0,
    block16 = 1,
    block32 = 2,
};

/// tcgen05 MMA Collector Buffer B Attribute.
pub const Tcgen05MMACollectorBBuffer = enum(u32) {
    b0 = 0,
    b1 = 1,
    b2 = 2,
    b3 = 3,
};

/// tcgen05.mma Collector Buffer Operation.
pub const Tcgen05MMACollectorOp = enum(u32) {
    discard = 0,
    lastuse = 1,
    fill = 2,
    use = 3,
};

/// tcgen05 MMA Supported Types.
pub const Tcgen05MMAKind = enum(u32) {
    f16 = 0,
    tf32 = 1,
    f8f6f4 = 2,
    i8 = 3,
    mxf8f6f4 = 4,
    mxf4 = 5,
    mxf4nvf4 = 6,
    ti16 = 7,
};

/// NVVM Tcgen05 wait kind.
pub const Tcgen05WaitKind = enum(u32) {
    load = 0,
    store = 1,
};

/// NVVM Tensormap Elemtype.
pub const TensormapElemtype = enum(u32) {
    u8 = 0,
    u16 = 1,
    u32 = 2,
    s32 = 3,
    u64 = 4,
    s64 = 5,
    f16 = 6,
    f32 = 7,
    @"f32.ftz" = 8,
    f64 = 9,
    bf16 = 10,
    tf32 = 11,
    @"tf32.ftz" = 12,
    b4x16 = 13,
    b4x16_p64 = 14,
    b6x16_p32 = 15,
};

/// NVVM Tensormap Field Kind.
pub const TensormapField = enum(u32) {
    global_address = 0,
    rank = 1,
    box_dim = 2,
    global_dim = 3,
    global_stride = 4,
    element_stride = 5,
    elemtype = 6,
    interleave_layout = 7,
    swizzle_mode = 8,
    swizzle_atomicity = 9,
    fill_mode = 10,
};

/// NVVM Tensormap Fill Mode.
pub const TensormapFillMode = enum(u32) {
    zero = 0,
    oob_nan = 1,
};

/// NVVM Tensormap Interleave Layout.
pub const TensormapInterleaveLayout = enum(u32) {
    no_interleave = 0,
    b16 = 1,
    b32 = 2,
};

/// NVVM Tensormap Swizzle Atomicity.
pub const TensormapSwizzleAtomicity = enum(u32) {
    b16 = 0,
    b32 = 1,
    b32_flip_b8 = 2,
    b64 = 3,
};

/// NVVM Tensormap Swizzle Mode.
pub const TensormapSwizzleMode = enum(u32) {
    no_swizzling = 0,
    b32 = 1,
    b64 = 2,
    b128 = 3,
    b96 = 4,
};

/// Tensor Memory Layout Enumerated Type.
pub const TmemLayout = enum(u32) {
    tmem_16dp_128bit = 0,
    tmem_16dp_256bit = 1,
    tmem_32dp_32bit = 2,
    tmem_16dp_64bit = 3,
    tmem_16dp_32bit_t0_t15 = 4,
    tmem_16dp_32bit_t16_t31 = 5,
};

/// NVVM validate data pattern.
pub const ValidatePattern = enum(u32) {
    none = 0,
    per_16bytes_80000000 = 1,
    per_16bytes_8000 = 2,
    per_16bytes_80 = 3,
    per_16bytes_8 = 4,
    per_element_ff = 5,
    per_element_80000000 = 6,
    per_element_8000 = 7,
    per_element_80 = 8,
    per_element_8 = 9,
    per_16bytes_80000000_paired = 10,
};

/// NVVM vote sync kind.
pub const VoteSyncKind = enum(u32) {
    any = 0,
    all = 1,
    ballot = 2,
    uni = 3,
};

/// WGMMA overflow options.
pub const WGMMAScaleIn = enum(u32) {
    one = 1,
    neg = 2,
};

/// WGMMA input predicate.
pub const WGMMAScaleOut = enum(u32) {
    zero = 0,
    one = 1,
};

/// NVVM WGMMA types.
pub const WGMMATypes = enum(u32) {
    f16 = 0,
    tf32 = 1,
    u8 = 2,
    s8 = 3,
    b1 = 4,
    bf16 = 5,
    e4m3 = 6,
    e5m2 = 7,
    f32 = 8,
    s32 = 9,
};

// Types and attributes, generated from the dialect's .td files.

/// `#nvvm.atomic_op`: operations supported by atom instruction
pub const AtomicOpKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMAtomicOpKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atomic_op";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: AtomicOpKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMAtomicOpKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) AtomicOpKind {
        return @enumFromInt(c.mlirCuteNVVMAtomicOpKindAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.reduction`: NVVM barrier reduction operation
pub const BarrierReductionAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMBarrierReduction;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "reduction";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: BarrierReduction,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMBarrierReductionAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) BarrierReduction {
        return @enumFromInt(c.mlirCuteNVVMBarrierReductionAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.barrier_redux_kind`: NVVM barrier redux kind
pub const BarrierReduxKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMBarrierReduxKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "barrier_redux_kind";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: BarrierReduxKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMBarrierReduxKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) BarrierReduxKind {
        return @enumFromInt(c.mlirCuteNVVMBarrierReduxKindAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.block_scale_format`: MMA Block Scale Format
pub const BlockScaleFormatAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMBlockScaleFormat;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "block_scale_format";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: BlockScaleFormat,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMBlockScaleFormatAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) BlockScaleFormat {
        return @enumFromInt(c.mlirCuteNVVMBlockScaleFormatAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.cta_group`: NVVM CTA group kind
pub const CTAGroupKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMCTAGroupKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "cta_group";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: CTAGroupKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMCTAGroupKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) CTAGroupKind {
        return @enumFromInt(c.mlirCuteNVVMCTAGroupKindAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.packfloat_type`: NVVM CVT Pack Float kind
pub const CVTPackFloatKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMCVTPackFloatKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "packfloat_type";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: CVTPackFloatKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMCVTPackFloatKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) CVTPackFloatKind {
        return @enumFromInt(c.mlirCuteNVVMCVTPackFloatKindAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.cache_eviction_priority`: NVVM Cache Eviction Priority
pub const CacheEvictionPriorityAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMCacheEvictionPriority;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "cache_eviction_priority";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: CacheEvictionPriority,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMCacheEvictionPriorityAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) CacheEvictionPriority {
        return @enumFromInt(c.mlirCuteNVVMCacheEvictionPriorityAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.cluster_launch_control_query_type`: NVVM ClusterLaunchControlQueryType
pub const ClusterLaunchControlQueryTypeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMClusterLaunchControlQueryType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "cluster_launch_control_query_type";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: ClusterLaunchControlQueryType,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMClusterLaunchControlQueryTypeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) ClusterLaunchControlQueryType {
        return @enumFromInt(c.mlirCuteNVVMClusterLaunchControlQueryTypeAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.op`: Comparison operator encoding
pub const CompareOpKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMCompareOpKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "op";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: CompareOpKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMCompareOpKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) CompareOpKind {
        return @enumFromInt(c.mlirCuteNVVMCompareOpKindAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.convert_fp4_type`: NVVM ConvertFP4Type kind
pub const ConvertFP4TypeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMConvertFP4Type;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "convert_fp4_type";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: ConvertFP4Type,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMConvertFP4TypeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) ConvertFP4Type {
        return @enumFromInt(c.mlirCuteNVVMConvertFP4TypeAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.convert_fp8_type`: NVVM ConvertFP8Type kind
pub const ConvertFP8TypeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMConvertFP8Type;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "convert_fp8_type";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: ConvertFP8Type,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMConvertFP8TypeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) ConvertFP8Type {
        return @enumFromInt(c.mlirCuteNVVMConvertFP8TypeAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.convert_scale_kind`: NVVM ConvertScale kind
pub const ConvertScaleKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMConvertScaleKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "convert_scale_kind";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: ConvertScaleKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMConvertScaleKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) ConvertScaleKind {
        return @enumFromInt(c.mlirCuteNVVMConvertScaleKindAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.dot_accumulate_type`: NVVM DotAccumulateType
pub const DotAccumulateTypeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMDotAccumulateType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "dot_accumulate_type";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: DotAccumulateType,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMDotAccumulateTypeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) DotAccumulateType {
        return @enumFromInt(c.mlirCuteNVVMDotAccumulateTypeAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.evict_kind`: NVVM L2 Prefetch Size
pub const EvictKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMEvictKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "evict_kind";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: EvictKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMEvictKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) EvictKind {
        return @enumFromInt(c.mlirCuteNVVMEvictKindAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.fp_rnd_mode`: NVVM FPRoundingMode kind
pub const FPRoundingModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMFPRoundingMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "fp_rnd_mode";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: FPRoundingMode,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMFPRoundingModeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) FPRoundingMode {
        return @enumFromInt(c.mlirCuteNVVMFPRoundingModeAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.grid_dep_action`: Action kind for grid dependency control
pub const GridDepActionKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMGridDepActionKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "grid_dep_action";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: GridDepActionKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMGridDepActionKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) GridDepActionKind {
        return @enumFromInt(c.mlirCuteNVVMGridDepActionKindAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.int_rnd_mode`: NVVM IntegerRoundingMode kind
pub const IntegerRoundingModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMIntegerRoundingMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "int_rnd_mode";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: IntegerRoundingMode,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMIntegerRoundingModeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) IntegerRoundingMode {
        return @enumFromInt(c.mlirCuteNVVMIntegerRoundingModeAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.l2_prefetch`: NVVM L2 Prefetch Size
pub const L2PrefetchSizeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVML2PrefetchSize;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "l2_prefetch";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: L2PrefetchSize,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVML2PrefetchSizeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) L2PrefetchSize {
        return @enumFromInt(c.mlirCuteNVVML2PrefetchSizeAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.ld_st_matrix_elt_type`: Element type for ldmatrix and stmatrix
pub const LdStMatrixEltTypeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMLdStMatrixEltType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "ld_st_matrix_elt_type";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: LdStMatrixEltType,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMLdStMatrixEltTypeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) LdStMatrixEltType {
        return @enumFromInt(c.mlirCuteNVVMLdStMatrixEltTypeAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.load_cache_modifier_ext`: NVVM load cache modifier kind(Ext)
pub const LoadCacheModifierExtKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMLoadCacheModifierExtKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "load_cache_modifier_ext";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: LoadCacheModifierExtKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMLoadCacheModifierExtKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) LoadCacheModifierExtKind {
        return @enumFromInt(c.mlirCuteNVVMLoadCacheModifierExtKindAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.load_cache_modifier`: NVVM load cache modifier kind
pub const LoadCacheModifierKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMLoadCacheModifierKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "load_cache_modifier";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: LoadCacheModifierKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMLoadCacheModifierKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) LoadCacheModifierKind {
        return @enumFromInt(c.mlirCuteNVVMLoadCacheModifierKindAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.load_shape`: shape attribute for ldmatrix
pub const LoadShapeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMLoadShape;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "load_shape";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: LoadShape,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMLoadShapeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) LoadShape {
        return @enumFromInt(c.mlirCuteNVVMLoadShapeAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.load_src_format`: source format for ldmatrix
pub const LoadSrcFormatAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMLoadSrcFormat;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "load_src_format";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: LoadSrcFormat,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMLoadSrcFormatAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) LoadSrcFormat {
        return @enumFromInt(c.mlirCuteNVVMLoadSrcFormatAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.mbarrier_layout`: NVVM MBarrier Layout
pub const MBarrierLayoutAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMBarrierLayout;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "mbarrier_layout";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MBarrierLayout,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMBarrierLayoutAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) MBarrierLayout {
        return @enumFromInt(c.mlirCuteNVVMMBarrierLayoutAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.mbarrier_phase`: NVVM mbarrier phase type
pub const MBarrierPhaseAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMBarrierPhase;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "mbarrier_phase";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MBarrierPhase,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMBarrierPhaseAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) MBarrierPhase {
        return @enumFromInt(c.mlirCuteNVVMMBarrierPhaseAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.mbar_scope`: NVVM MBarrier scope kind
pub const MBarrierScopeKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMBarrierScopeKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "mbar_scope";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MBarrierScopeKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMBarrierScopeKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) MBarrierScopeKind {
        return @enumFromInt(c.mlirCuteNVVMMBarrierScopeKindAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.mbar_space`: NVVM MBarrier space kind
pub const MBarrierSpaceKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMBarrierSpaceKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "mbar_space";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MBarrierSpaceKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMBarrierSpaceKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) MBarrierSpaceKind {
        return @enumFromInt(c.mlirCuteNVVMMBarrierSpaceKindAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.mbar_txn_kind`: NVVM MBarrier Transaction kind
pub const MBarrierTxnKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMBarrierTxnKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "mbar_txn_kind";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MBarrierTxnKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMBarrierTxnKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) MBarrierTxnKind {
        return @enumFromInt(c.mlirCuteNVVMMBarrierTxnKindAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.mbar_wait`: NVVM MBarrier wait kind
pub const MBarrierWaitKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMBarrierWaitKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "mbar_wait";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MBarrierWaitKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMBarrierWaitKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) MBarrierWaitKind {
        return @enumFromInt(c.mlirCuteNVVMMBarrierWaitKindAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.mma_b1op`: MMA binary operations
pub const MMAB1OpAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMMAB1Op;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "mma_b1op";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MMAB1Op,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMMAB1OpAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) MMAB1Op {
        return @enumFromInt(c.mlirCuteNVVMMMAB1OpAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.block_scale_kind`: Block Scale Kind
pub const MMABlockScaleKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMMABlockScaleKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "block_scale_kind";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MMABlockScaleKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMMABlockScaleKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) MMABlockScaleKind {
        return @enumFromInt(c.mlirCuteNVVMMMABlockScaleKindAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.mma_cta_count`: MMA CTA count
pub const MMACtaCountAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMMACtaCount;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "mma_cta_count";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MMACtaCount,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMMACtaCountAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) MMACtaCount {
        return @enumFromInt(c.mlirCuteNVVMMMACtaCountAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.mma_frag`: NVVM MMA frag type
pub const MMAFragAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMMAFrag;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "mma_frag";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MMAFrag,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMMAFragAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) MMAFrag {
        return @enumFromInt(c.mlirCuteNVVMMMAFragAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.mma_int_overflow`: MMA overflow options
pub const MMAIntOverflowAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMMAIntOverflow;
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
        const result = c.mlirCuteNVVMMMAIntOverflowAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) MMAIntOverflow {
        return @enumFromInt(c.mlirCuteNVVMMMAIntOverflowAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.mma_kind`: MMA operation kind
pub const MMAKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMMAKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "mma_kind";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MMAKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMMAKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) MMAKind {
        return @enumFromInt(c.mlirCuteNVVMMMAKindAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.mma_layout`: NVVM MMA layout
pub const MMALayoutAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMMALayout;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "mma_layout";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MMALayout,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMMALayoutAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) MMALayout {
        return @enumFromInt(c.mlirCuteNVVMMMALayoutAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.mma_type`: NVVM MMA types
pub const MMATypesAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMMATypes;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "mma_type";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MMATypes,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMMATypesAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) MMATypes {
        return @enumFromInt(c.mlirCuteNVVMMMATypesAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.match_sync_kind`: NVVM match sync kind
pub const MatchSyncKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMatchSyncKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "match_sync_kind";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MatchSyncKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMatchSyncKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) MatchSyncKind {
        return @enumFromInt(c.mlirCuteNVVMMatchSyncKindAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.mem_order`: NVVM Memory Ordering kind
pub const MemOrderKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMemOrderKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "mem_order";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MemOrderKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMemOrderKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) MemOrderKind {
        return @enumFromInt(c.mlirCuteNVVMMemOrderKindAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.mem_scope`: NVVM Memory Scope kind
pub const MemScopeKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMemScopeKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "mem_scope";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MemScopeKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMemScopeKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) MemScopeKind {
        return @enumFromInt(c.mlirCuteNVVMMemScopeKindAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.mul_mode`: multiply mode attribute
pub const MulModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMulMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "mul_mode";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MulMode,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMulModeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) MulMode {
        return @enumFromInt(c.mlirCuteNVVMMulModeAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.memory_space`: NVVM Memory Space
pub const NVVMMemorySpaceAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMNVVMMemorySpace;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "memory_space";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: NVVMMemorySpace,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMNVVMMemorySpaceAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) NVVMMemorySpace {
        return @enumFromInt(c.mlirCuteNVVMNVVMMemorySpaceAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.permute_mode`: NVVM permute mode
pub const PermuteModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMPermuteMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "permute_mode";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: PermuteMode,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMPermuteModeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) PermuteMode {
        return @enumFromInt(c.mlirCuteNVVMPermuteModeAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.prefetch_cache_level`: NVVM Prefetch Cache Level
pub const PrefetchCacheLevelAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMPrefetchCacheLevel;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "prefetch_cache_level";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: PrefetchCacheLevel,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMPrefetchCacheLevelAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) PrefetchCacheLevel {
        return @enumFromInt(c.mlirCuteNVVMPrefetchCacheLevelAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.proxy_kind`: Proxy kind
pub const ProxyKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMProxyKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "proxy_kind";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: ProxyKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMProxyKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) ProxyKind {
        return @enumFromInt(c.mlirCuteNVVMProxyKindAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.reduction_kind`: NVVM Reduction Kind attribute
pub const ReductionKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMReductionKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "reduction_kind";
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
        const result = c.mlirCuteNVVMReductionKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) ReductionKind {
        return @enumFromInt(c.mlirCuteNVVMReductionKindAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.red_op`: Ops supported by red instruction
pub const ReductionOpAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMReductionOp;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "red_op";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: ReductionOp,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMReductionOpAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) ReductionOp {
        return @enumFromInt(c.mlirCuteNVVMReductionOpAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.red_type`: types supported by red instruction
pub const ReductionTypeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMReductionType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "red_type";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: ReductionType,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMReductionTypeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) ReductionType {
        return @enumFromInt(c.mlirCuteNVVMReductionTypeAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.spcomp_elem_size`: Sparse tensor compression element size
pub const SPCompressElemSizeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMSPCompressElemSize;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "spcomp_elem_size";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: SPCompressElemSize,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMSPCompressElemSizeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) SPCompressElemSize {
        return @enumFromInt(c.mlirCuteNVVMSPCompressElemSizeAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.spcomp_factor`: Sparse tensor compression factor type
pub const SPCompressFactorTypeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMSPCompressFactorType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "spcomp_factor";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: SPCompressFactorType,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMSPCompressFactorTypeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) SPCompressFactorType {
        return @enumFromInt(c.mlirCuteNVVMSPCompressFactorTypeAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.spcomp_index_size`: Sparse tensor compression index size
pub const SPCompressIndexSizeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMSPCompressIndexSize;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "spcomp_index_size";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: SPCompressIndexSize,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMSPCompressIndexSizeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) SPCompressIndexSize {
        return @enumFromInt(c.mlirCuteNVVMSPCompressIndexSizeAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.spcomp_op_kind`: spcompress operation kind
pub const SPCompressOpKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMSPCompressOpKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "spcomp_op_kind";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: SPCompressOpKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMSPCompressOpKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) SPCompressOpKind {
        return @enumFromInt(c.mlirCuteNVVMSPCompressOpKindAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.spcomp_rep_factor`: Sparse tensor compression repetition factor
pub const SPCompressRepFactorAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMSPCompressRepFactor;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "spcomp_rep_factor";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: SPCompressRepFactor,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMSPCompressRepFactorAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) SPCompressRepFactor {
        return @enumFromInt(c.mlirCuteNVVMSPCompressRepFactorAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.spdecomp_elem_size`: Sparse tensor decompression element size
pub const SPDecompressElemSizeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMSPDecompressElemSize;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "spdecomp_elem_size";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: SPDecompressElemSize,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMSPDecompressElemSizeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) SPDecompressElemSize {
        return @enumFromInt(c.mlirCuteNVVMSPDecompressElemSizeAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.spdecomp_factor`: Sparse tensor decompression factor type
pub const SPDecompressFactorTypeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMSPDecompressFactorType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "spdecomp_factor";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: SPDecompressFactorType,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMSPDecompressFactorTypeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) SPDecompressFactorType {
        return @enumFromInt(c.mlirCuteNVVMSPDecompressFactorTypeAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.spdecomp_index_size`: Sparse tensor decompression index size
pub const SPDecompressIndexSizeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMSPDecompressIndexSize;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "spdecomp_index_size";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: SPDecompressIndexSize,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMSPDecompressIndexSizeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) SPDecompressIndexSize {
        return @enumFromInt(c.mlirCuteNVVMSPDecompressIndexSizeAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.spdecomp_rep_factor`: Sparse tensor decompression repetition factor
pub const SPDecompressRepFactorAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMSPDecompressRepFactor;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "spdecomp_rep_factor";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: SPDecompressRepFactor,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMSPDecompressRepFactorAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) SPDecompressRepFactor {
        return @enumFromInt(c.mlirCuteNVVMSPDecompressRepFactorAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.sat_mode`: NVVM SaturationMode kind
pub const SaturationModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMSaturationMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sat_mode";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: SaturationMode,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMSaturationModeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) SaturationMode {
        return @enumFromInt(c.mlirCuteNVVMSaturationModeAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.sat`: NVVM SaturationMode kind
pub const SaturationModeKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMSaturationModeKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sat";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: SaturationModeKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMSaturationModeKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) SaturationModeKind {
        return @enumFromInt(c.mlirCuteNVVMSaturationModeKindAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.scale_vec_size`: MMA Scale Vector Sizes
pub const ScaleVecSizeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMScaleVecSize;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "scale_vec_size";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: ScaleVecSize,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMScaleVecSizeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) ScaleVecSize {
        return @enumFromInt(c.mlirCuteNVVMScaleVecSizeAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.action`: NVVM set max register action
pub const SetMaxRegisterActionAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMSetMaxRegisterAction;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "action";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: SetMaxRegisterAction,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMSetMaxRegisterActionAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) SetMaxRegisterAction {
        return @enumFromInt(c.mlirCuteNVVMSetMaxRegisterActionAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.shared_space`: Shared memory space
pub const SharedSpaceAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMSharedSpace;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "shared_space";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: SharedSpace,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMSharedSpaceAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) SharedSpace {
        return @enumFromInt(c.mlirCuteNVVMSharedSpaceAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.shfl_kind`: NVVM shuffle kind
pub const ShflKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMShflKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "shfl_kind";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: ShflKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMShflKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) ShflKind {
        return @enumFromInt(c.mlirCuteNVVMShflKindAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.sparsity_format`: MMA Sparsity Format
pub const SparsityFormatAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMSparsityFormat;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sparsity_format";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: SparsityFormat,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMSparsityFormatAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) SparsityFormat {
        return @enumFromInt(c.mlirCuteNVVMSparsityFormatAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.state_space`: NVVM State Space
pub const StateSpaceAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMStateSpace;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "state_space";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: StateSpace,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMStateSpaceAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) StateSpace {
        return @enumFromInt(c.mlirCuteNVVMStateSpaceAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.store_cache_modifier`: NVVM store cache modifier kind
pub const StoreCacheModifierKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMStoreCacheModifierKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "store_cache_modifier";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: StoreCacheModifierKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMStoreCacheModifierKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) StoreCacheModifierKind {
        return @enumFromInt(c.mlirCuteNVVMStoreCacheModifierKindAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.store_shape`: shape attribute for stmatrix
pub const StoreShapeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMStoreShape;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "store_shape";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: StoreShape,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMStoreShapeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) StoreShape {
        return @enumFromInt(c.mlirCuteNVVMStoreShapeAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.TCBarParam`: Cluster MMA Barrier Parameter Type
pub const TCBarParamAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTCBarParam;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "TCBarParam";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: TCBarParam,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTCBarParamAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) TCBarParam {
        return @enumFromInt(c.mlirCuteNVVMTCBarParamAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.tma_load_mode`: NVVM TMA Load Mode
pub const TMALoadModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTMALoadMode;
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
        value: TMALoadMode,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTMALoadModeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) TMALoadMode {
        return @enumFromInt(c.mlirCuteNVVMTMALoadModeAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.tma_redux_kind`: NVVM TMA redux kind
pub const TMAReduxKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTMAReduxKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tma_redux_kind";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: TMAReduxKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTMAReduxKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) TMAReduxKind {
        return @enumFromInt(c.mlirCuteNVVMTMAReduxKindAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.tma_store_mode`: NVVM TMA Store Mode
pub const TMAStoreModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTMAStoreMode;
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
        value: TMAStoreMode,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTMAStoreModeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) TMAStoreMode {
        return @enumFromInt(c.mlirCuteNVVMTMAStoreModeAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.tcgen05_cp_multicast`: tcgen05 cp multicast
pub const Tcgen05CpMulticastAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTcgen05CpMulticast;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tcgen05_cp_multicast";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: Tcgen05CpMulticast,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTcgen05CpMulticastAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) Tcgen05CpMulticast {
        return @enumFromInt(c.mlirCuteNVVMTcgen05CpMulticastAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.tcgen05_cp_shape`: tcgen05 cp shapes
pub const Tcgen05CpShapeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTcgen05CpShape;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tcgen05_cp_shape";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: Tcgen05CpShape,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTcgen05CpShapeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) Tcgen05CpShape {
        return @enumFromInt(c.mlirCuteNVVMTcgen05CpShapeAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.tcgen05_cp_src_fmt`: tcgen05 cp source format
pub const Tcgen05CpSrcFormatAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTcgen05CpSrcFormat;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tcgen05_cp_src_fmt";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: Tcgen05CpSrcFormat,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTcgen05CpSrcFormatAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) Tcgen05CpSrcFormat {
        return @enumFromInt(c.mlirCuteNVVMTcgen05CpSrcFormatAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.tcgen05_fence`: NVVM Tcgen05 fence kind
pub const Tcgen05FenceKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTcgen05FenceKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tcgen05_fence";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: Tcgen05FenceKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTcgen05FenceKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) Tcgen05FenceKind {
        return @enumFromInt(c.mlirCuteNVVMTcgen05FenceKindAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.tcgen05_ldst_shape`: allowed 32-bit signless integer cases: 0, 1, 2, 3, 4
pub const Tcgen05LdStShapeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTcgen05LdStShape;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tcgen05_ldst_shape";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: Tcgen05LdStShape,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTcgen05LdStShapeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) Tcgen05LdStShape {
        return @enumFromInt(c.mlirCuteNVVMTcgen05LdStShapeAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.tcgen05_mma_block_scale`: tcgen05.mma block scale attribute
pub const Tcgen05MMABlockScaleAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTcgen05MMABlockScale;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tcgen05_mma_block_scale";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: Tcgen05MMABlockScale,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTcgen05MMABlockScaleAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) Tcgen05MMABlockScale {
        return @enumFromInt(c.mlirCuteNVVMTcgen05MMABlockScaleAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.tcgen05_mma_collectorb`: tcgen05 MMA Collector Buffer B Attribute
pub const Tcgen05MMACollectorBBufferAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTcgen05MMACollectorBBuffer;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tcgen05_mma_collectorb";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: Tcgen05MMACollectorBBuffer,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTcgen05MMACollectorBBufferAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) Tcgen05MMACollectorBBuffer {
        return @enumFromInt(c.mlirCuteNVVMTcgen05MMACollectorBBufferAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.tcgen05_mma_collectorop`: tcgen05.mma Collector Buffer Operation
pub const Tcgen05MMACollectorOpAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTcgen05MMACollectorOp;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tcgen05_mma_collectorop";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: Tcgen05MMACollectorOp,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTcgen05MMACollectorOpAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) Tcgen05MMACollectorOp {
        return @enumFromInt(c.mlirCuteNVVMTcgen05MMACollectorOpAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.tcgen05_mma_kind`: tcgen05 MMA Supported Types
pub const Tcgen05MMAKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTcgen05MMAKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tcgen05_mma_kind";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: Tcgen05MMAKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTcgen05MMAKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) Tcgen05MMAKind {
        return @enumFromInt(c.mlirCuteNVVMTcgen05MMAKindAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.tcgen05_wait`: NVVM Tcgen05 wait kind
pub const Tcgen05WaitKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTcgen05WaitKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tcgen05_wait";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: Tcgen05WaitKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTcgen05WaitKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) Tcgen05WaitKind {
        return @enumFromInt(c.mlirCuteNVVMTcgen05WaitKindAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.tensormap_elemtype`: NVVM Tensormap Elemtype
pub const TensormapElemtypeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTensormapElemtype;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tensormap_elemtype";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: TensormapElemtype,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTensormapElemtypeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) TensormapElemtype {
        return @enumFromInt(c.mlirCuteNVVMTensormapElemtypeAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.tensormap_field`: NVVM Tensormap Field Kind
pub const TensormapFieldAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTensormapField;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tensormap_field";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: TensormapField,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTensormapFieldAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) TensormapField {
        return @enumFromInt(c.mlirCuteNVVMTensormapFieldAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.tensormap_fill_mode`: NVVM Tensormap Fill Mode
pub const TensormapFillModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTensormapFillMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tensormap_fill_mode";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: TensormapFillMode,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTensormapFillModeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) TensormapFillMode {
        return @enumFromInt(c.mlirCuteNVVMTensormapFillModeAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.tensormap_interleave_layout`: NVVM Tensormap Interleave Layout
pub const TensormapInterleaveLayoutAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTensormapInterleaveLayout;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tensormap_interleave_layout";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: TensormapInterleaveLayout,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTensormapInterleaveLayoutAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) TensormapInterleaveLayout {
        return @enumFromInt(c.mlirCuteNVVMTensormapInterleaveLayoutAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.tensormap_swizzle_atomicity`: NVVM Tensormap Swizzle Atomicity
pub const TensormapSwizzleAtomicityAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTensormapSwizzleAtomicity;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tensormap_swizzle_atomicity";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: TensormapSwizzleAtomicity,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTensormapSwizzleAtomicityAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) TensormapSwizzleAtomicity {
        return @enumFromInt(c.mlirCuteNVVMTensormapSwizzleAtomicityAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.tensormap_swizzle_mode`: NVVM Tensormap Swizzle Mode
pub const TensormapSwizzleModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTensormapSwizzleMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tensormap_swizzle_mode";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: TensormapSwizzleMode,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTensormapSwizzleModeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) TensormapSwizzleMode {
        return @enumFromInt(c.mlirCuteNVVMTensormapSwizzleModeAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.TmemLayout`: Tensor Memory Layout Enumerated Type
pub const TmemLayoutAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTmemLayout;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "TmemLayout";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: TmemLayout,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTmemLayoutAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) TmemLayout {
        return @enumFromInt(c.mlirCuteNVVMTmemLayoutAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.validate_pattern`: NVVM validate data pattern
pub const ValidatePatternAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMValidatePattern;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "validate_pattern";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: ValidatePattern,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMValidatePatternAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) ValidatePattern {
        return @enumFromInt(c.mlirCuteNVVMValidatePatternAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.vote_sync_kind`: NVVM vote sync kind
pub const VoteSyncKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMVoteSyncKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "vote_sync_kind";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: VoteSyncKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMVoteSyncKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) VoteSyncKind {
        return @enumFromInt(c.mlirCuteNVVMVoteSyncKindAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.wgmma_scale_in`: WGMMA overflow options
pub const WGMMAScaleInAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMWGMMAScaleIn;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "wgmma_scale_in";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: WGMMAScaleIn,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMWGMMAScaleInAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) WGMMAScaleIn {
        return @enumFromInt(c.mlirCuteNVVMWGMMAScaleInAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.wgmma_scale_out`: WGMMA input predicate
pub const WGMMAScaleOutAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMWGMMAScaleOut;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "wgmma_scale_out";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: WGMMAScaleOut,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMWGMMAScaleOutAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) WGMMAScaleOut {
        return @enumFromInt(c.mlirCuteNVVMWGMMAScaleOutAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.wgmma_type`: NVVM WGMMA types
pub const WGMMATypesAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMWGMMATypes;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "wgmma_type";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: WGMMATypes,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMWGMMATypesAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) WGMMATypes {
        return @enumFromInt(c.mlirCuteNVVMWGMMATypesAttrGetValue(self.ptr()));
    }
};

/// `#nvvm.ld_st_matrix_shape`: Matrix shape of ldmatrix, stmatrix and movmatrix
pub const LdStMatrixShapeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMLdStMatrixShape;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "ld_st_matrix_shape";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        m: c_int,
        n: c_int,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMLdStMatrixShapeAttrGet(ctx.ptr(), args.m, args.n);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getM(self: *const Self) c_int {
        return c.mlirCuteNVVMLdStMatrixShapeAttrGetM(self.ptr());
    }
    pub fn getN(self: *const Self) c_int {
        return c.mlirCuteNVVMLdStMatrixShapeAttrGetN(self.ptr());
    }
};

/// `#nvvm.shape`: Shape of an MMA operation
pub const MMAShapeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMMAShape;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "shape";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        m: c_int,
        n: c_int,
        k: c_int,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMMAShapeAttrGet(ctx.ptr(), args.m, args.n, args.k);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getM(self: *const Self) c_int {
        return c.mlirCuteNVVMMMAShapeAttrGetM(self.ptr());
    }
    pub fn getN(self: *const Self) c_int {
        return c.mlirCuteNVVMMMAShapeAttrGetN(self.ptr());
    }
    pub fn getK(self: *const Self) c_int {
        return c.mlirCuteNVVMMMAShapeAttrGetK(self.ptr());
    }
};

/// `#nvvm.target`: GPU target of an NVVM module
pub const TargetAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTarget;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "target";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        O: c_int = 2,
        triple: []const u8 = "nvptx64-nvidia-cuda",
        chip: []const u8 = "sm_75",
        features: []const u8 = "",
        flags: ?*const mlir.Attribute = null,
        link: ?*const mlir.Attribute = null,
        verifyTarget: bool = false,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTargetAttrGet(ctx.ptr(), args.O, mlir.stringRef(args.triple), mlir.stringRef(args.chip), mlir.stringRef(args.features), if (args.flags) |v| v.ptr() else .{ .ptr = null }, if (args.link) |v| v.ptr() else .{ .ptr = null }, args.verifyTarget);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getO(self: *const Self) c_int {
        return c.mlirCuteNVVMTargetAttrGetO(self.ptr());
    }
    pub fn getTriple(self: *const Self) []const u8 {
        const s = c.mlirCuteNVVMTargetAttrGetTriple(self.ptr());
        return if (s.length == 0) "" else mlir.string(s);
    }
    pub fn getChip(self: *const Self) []const u8 {
        const s = c.mlirCuteNVVMTargetAttrGetChip(self.ptr());
        return if (s.length == 0) "" else mlir.string(s);
    }
    pub fn getFeatures(self: *const Self) []const u8 {
        const s = c.mlirCuteNVVMTargetAttrGetFeatures(self.ptr());
        return if (s.length == 0) "" else mlir.string(s);
    }
    pub fn getFlags(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVVMTargetAttrGetFlags(self.ptr()).ptr);
    }
    pub fn getLink(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVVMTargetAttrGetLink(self.ptr()).ptr);
    }
    pub fn getVerifyTarget(self: *const Self) bool {
        return c.mlirCuteNVVMTargetAttrGetVerifyTarget(self.ptr());
    }
};

// Operation builders, generated from the operations' .td.
/// `nvvm.add.packed.bf16x2.f32x2.f32x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn add_packed_bf16x2_f32x2_f32x2(ctx: *mlir.Context, srcA: *const mlir.Value, srcB: *const mlir.Value, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.add.packed.bf16x2.f32x2.f32x2", .{
        .operands = .{ .flat = &.{ srcA, srcB } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.add.packed.f16x2.f32x2.f32x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn add_packed_f16x2_f32x2_f32x2(ctx: *mlir.Context, srcA: *const mlir.Value, srcB: *const mlir.Value, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.add.packed.f16x2.f32x2.f32x2", .{
        .operands = .{ .flat = &.{ srcA, srcB } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.add.packed.f32x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn add_packed_f32x2(ctx: *mlir.Context, srcA: *const mlir.Value, srcB: *const mlir.Value, res_type: *const mlir.Type, rnd: ?*const mlir.Attribute, ftz: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (ftz) |value| attributes.appendAssumeCapacity(.named(ctx, "ftz", value));
    return mlir.Operation.make(ctx, "nvvm.add.packed.f32x2", .{
        .operands = .{ .flat = &.{ srcA, srcB } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.add.packed.f32x2.bf16x2.f32x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn add_packed_f32x2_bf16x2_f32x2(ctx: *mlir.Context, srcA: *const mlir.Value, srcB: *const mlir.Value, res_type: *const mlir.Type, rnd: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    return mlir.Operation.make(ctx, "nvvm.add.packed.f32x2.bf16x2.f32x2", .{
        .operands = .{ .flat = &.{ srcA, srcB } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.add.packed.f32x2.f16x2.f32x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn add_packed_f32x2_f16x2_f32x2(ctx: *mlir.Context, srcA: *const mlir.Value, srcB: *const mlir.Value, res_type: *const mlir.Type, rnd: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    return mlir.Operation.make(ctx, "nvvm.add.packed.f32x2.f16x2.f32x2", .{
        .operands = .{ .flat = &.{ srcA, srcB } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.addf`. Result types are explicit; attributes use mlir.Attribute.
pub fn addf(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, res_type: *const mlir.Type, rnd: ?*const mlir.Attribute, sat: ?*const mlir.Attribute, ftz: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (sat) |value| attributes.appendAssumeCapacity(.named(ctx, "sat", value));
    if (ftz) |value| attributes.appendAssumeCapacity(.named(ctx, "ftz", value));
    return mlir.Operation.make(ctx, "nvvm.addf", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.applypriority.async.bulk`. Result types are explicit; attributes use mlir.Attribute.
pub fn applypriority_async_bulk(ctx: *mlir.Context, ptr: *const mlir.Value, size: *const mlir.Value, evict: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (evict) |value| attributes.appendAssumeCapacity(.named(ctx, "evict", value));
    return mlir.Operation.make(ctx, "nvvm.applypriority.async.bulk", .{
        .operands = .{ .flat = &.{ ptr, size } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.applypriority.async.bulk.tensor`. Result types are explicit; attributes use mlir.Attribute.
pub fn applypriority_async_bulk_tensor(ctx: *mlir.Context, tmaDesc: *const mlir.Value, coordinates: []const *const mlir.Value, im2colArgs: []const *const mlir.Value, mode: ?*const mlir.Attribute, evict: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (mode) |value| attributes.appendAssumeCapacity(.named(ctx, "mode", value));
    if (evict) |value| attributes.appendAssumeCapacity(.named(ctx, "evict", value));
    return mlir.Operation.make(ctx, "nvvm.applypriority.async.bulk.tensor", .{
        .operands = .{ .variadic = &.{
            &.{tmaDesc},
            coordinates,
            im2colArgs,
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.applypriority.async.bulk.tensor.override`. Result types are explicit; attributes use mlir.Attribute.
pub fn applypriority_async_bulk_tensor_override(ctx: *mlir.Context, tmaDesc: *const mlir.Value, overrideAdrr: *const mlir.Value, coordinates: []const *const mlir.Value, tensorSize: []const *const mlir.Value, lowerStride: []const *const mlir.Value, upperStride: ?*const mlir.Value, im2colArgs: []const *const mlir.Value, mode: ?*const mlir.Attribute, evict: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (mode) |value| attributes.appendAssumeCapacity(.named(ctx, "mode", value));
    if (evict) |value| attributes.appendAssumeCapacity(.named(ctx, "evict", value));
    return mlir.Operation.make(ctx, "nvvm.applypriority.async.bulk.tensor.override", .{
        .operands = .{ .variadic = &.{
            &.{tmaDesc},
            &.{overrideAdrr},
            coordinates,
            tensorSize,
            lowerStride,
            if (upperStride) |value| &.{value} else &.{},
            im2colArgs,
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.atomicrmw`. Result types are explicit; attributes use mlir.Attribute.
pub fn atomicrmw(ctx: *mlir.Context, ptr: *const mlir.Value, a: *const mlir.Value, b: ?*const mlir.Value, res_type: *const mlir.Type, op: *const mlir.Attribute, memOrder: ?*const mlir.Attribute, syncscope: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "op", op));
    if (memOrder) |value| attributes.appendAssumeCapacity(.named(ctx, "memOrder", value));
    if (syncscope) |value| attributes.appendAssumeCapacity(.named(ctx, "syncscope", value));
    return mlir.Operation.make(ctx, "nvvm.atomicrmw", .{
        .operands = .{ .flat = if (b) |value| &.{ ptr, a, value } else &.{ ptr, a } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.bar.warp.sync`. Result types are explicit; attributes use mlir.Attribute.
pub fn bar_warp_sync(ctx: *mlir.Context, mask: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.bar.warp.sync", .{
        .operands = .{ .flat = &.{mask} },
        .location = location,
    });
}

/// `nvvm.barrier`. Result types are explicit; attributes use mlir.Attribute.
pub fn barrier(ctx: *mlir.Context, barrierId: ?*const mlir.Value, numberOfThreads: ?*const mlir.Value, reductionPredicate: ?*const mlir.Value, res_type: ?*const mlir.Type, reductionOp: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var results: stdx.BoundedArray(*const mlir.Type, 256) = .empty;
    if (res_type) |value| results.appendAssumeCapacity(value);
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (reductionOp) |value| attributes.appendAssumeCapacity(.named(ctx, "reductionOp", value));
    return mlir.Operation.make(ctx, "nvvm.barrier", .{
        .operands = .{ .variadic = &.{
            if (barrierId) |value| &.{value} else &.{},
            if (numberOfThreads) |value| &.{value} else &.{},
            if (reductionPredicate) |value| &.{value} else &.{},
        } },
        .results = .{ .flat = results.constSlice() },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.barrier.arrive`. Result types are explicit; attributes use mlir.Attribute.
pub fn barrier_arrive(ctx: *mlir.Context, barrierId: ?*const mlir.Value, numberOfThreads: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.barrier.arrive", .{
        .operands = .{ .flat = if (barrierId) |value| &.{ value, numberOfThreads } else &.{numberOfThreads} },
        .location = location,
    });
}

/// `nvvm.barrier.cta.arrive`. Result types are explicit; attributes use mlir.Attribute.
pub fn barrier_cta_arrive(ctx: *mlir.Context, barrierId: *const mlir.Value, threadCount: *const mlir.Value, aligned: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (aligned) |value| attributes.appendAssumeCapacity(.named(ctx, "aligned", value));
    return mlir.Operation.make(ctx, "nvvm.barrier.cta.arrive", .{
        .operands = .{ .flat = &.{ barrierId, threadCount } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.barrier.cta.red`. Result types are explicit; attributes use mlir.Attribute.
pub fn barrier_cta_red(ctx: *mlir.Context, pred: *const mlir.Value, barrierId: *const mlir.Value, threadCount: ?*const mlir.Value, res_type: *const mlir.Type, kind: *const mlir.Attribute, aligned: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (aligned) |value| attributes.appendAssumeCapacity(.named(ctx, "aligned", value));
    return mlir.Operation.make(ctx, "nvvm.barrier.cta.red", .{
        .operands = .{ .flat = if (threadCount) |value| &.{ pred, barrierId, value } else &.{ pred, barrierId } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.barrier.cta.sync`. Result types are explicit; attributes use mlir.Attribute.
pub fn barrier_cta_sync(ctx: *mlir.Context, barrierId: *const mlir.Value, threadCount: ?*const mlir.Value, aligned: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (aligned) |value| attributes.appendAssumeCapacity(.named(ctx, "aligned", value));
    return mlir.Operation.make(ctx, "nvvm.barrier.cta.sync", .{
        .operands = .{ .flat = if (threadCount) |value| &.{ barrierId, value } else &.{barrierId} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.barrier0`. Result types are explicit; attributes use mlir.Attribute.
pub fn barrier0(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.barrier0", .{
        .location = location,
    });
}

/// `nvvm.breakpoint`. Result types are explicit; attributes use mlir.Attribute.
pub fn breakpoint(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.breakpoint", .{
        .location = location,
    });
}

/// `nvvm.clmad`. Result types are explicit; attributes use mlir.Attribute.
pub fn clmad(ctx: *mlir.Context, a: *const mlir.Value, b: *const mlir.Value, c_: *const mlir.Value, res_type: *const mlir.Type, high: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (high) |value| attributes.appendAssumeCapacity(.named(ctx, "high", value));
    return mlir.Operation.make(ctx, "nvvm.clmad", .{
        .operands = .{ .flat = &.{ a, b, c_ } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cluster.arrive`. Result types are explicit; attributes use mlir.Attribute.
pub fn cluster_arrive(ctx: *mlir.Context, aligned: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (aligned) |value| attributes.appendAssumeCapacity(.named(ctx, "aligned", value));
    return mlir.Operation.make(ctx, "nvvm.cluster.arrive", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cluster.arrive.relaxed`. Result types are explicit; attributes use mlir.Attribute.
pub fn cluster_arrive_relaxed(ctx: *mlir.Context, aligned: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (aligned) |value| attributes.appendAssumeCapacity(.named(ctx, "aligned", value));
    return mlir.Operation.make(ctx, "nvvm.cluster.arrive.relaxed", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cluster.wait`. Result types are explicit; attributes use mlir.Attribute.
pub fn cluster_wait(ctx: *mlir.Context, aligned: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (aligned) |value| attributes.appendAssumeCapacity(.named(ctx, "aligned", value));
    return mlir.Operation.make(ctx, "nvvm.cluster.wait", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.clusterlaunchcontrol.query.cancel`. Result types are explicit; attributes use mlir.Attribute.
pub fn clusterlaunchcontrol_query_cancel(ctx: *mlir.Context, try_cancel_response: *const mlir.Value, res_type: *const mlir.Type, query_type: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.clusterlaunchcontrol.query.cancel", .{
        .operands = .{ .flat = &.{try_cancel_response} },
        .results = .{ .flat = &.{res_type} },
        .attributes = &.{
            .named(ctx, "query_type", query_type),
        },
        .location = location,
    });
}

/// `nvvm.clusterlaunchcontrol.try.cancel`. Result types are explicit; attributes use mlir.Attribute.
pub fn clusterlaunchcontrol_try_cancel(ctx: *mlir.Context, smemAddress: *const mlir.Value, mbarrier: *const mlir.Value, multicast: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (multicast) |value| attributes.appendAssumeCapacity(.named(ctx, "multicast", value));
    return mlir.Operation.make(ctx, "nvvm.clusterlaunchcontrol.try.cancel", .{
        .operands = .{ .flat = &.{ smemAddress, mbarrier } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.compare.and.set`. Result types are explicit; attributes use mlir.Attribute.
pub fn compare_and_set(ctx: *mlir.Context, a: *const mlir.Value, b: *const mlir.Value, result_type: *const mlir.Type, op: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.compare.and.set", .{
        .operands = .{ .flat = &.{ a, b } },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "op", op),
        },
        .location = location,
    });
}

/// `nvvm.convert.and.pack.integer`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_and_pack_integer(ctx: *mlir.Context, srcA: *const mlir.Value, srcB: *const mlir.Value, srcC: ?*const mlir.Value, dst_type: *const mlir.Type, is_signed: ?*const mlir.Attribute, convert_type: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (is_signed) |value| attributes.appendAssumeCapacity(.named(ctx, "is_signed", value));
    attributes.appendAssumeCapacity(.named(ctx, "convert_type", convert_type));
    return mlir.Operation.make(ctx, "nvvm.convert.and.pack.integer", .{
        .operands = .{ .flat = if (srcC) |value| &.{ srcA, srcB, value } else &.{ srcA, srcB } },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.bf16x2.to.f4x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_bf16x2_to_f4x2(ctx: *mlir.Context, src: *const mlir.Value, scale_factor: ?*const mlir.Value, dst_type: *const mlir.Type, isPZO: ?*const mlir.Attribute, scale_factor_kind: ?*const mlir.Attribute, rnd: ?*const mlir.Attribute, relu: ?*const mlir.Attribute, dstTy: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    if (isPZO) |value| attributes.appendAssumeCapacity(.named(ctx, "isPZO", value));
    if (scale_factor_kind) |value| attributes.appendAssumeCapacity(.named(ctx, "scale_factor_kind", value));
    if (rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    attributes.appendAssumeCapacity(.named(ctx, "dstTy", dstTy));
    return mlir.Operation.make(ctx, "nvvm.convert.bf16x2.to.f4x2", .{
        .operands = .{ .flat = if (scale_factor) |value| &.{ src, value } else &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.bf16x2.to.f6x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_bf16x2_to_f6x2(ctx: *mlir.Context, src: *const mlir.Value, scale_factor: ?*const mlir.Value, dst_type: *const mlir.Type, isPZO: ?*const mlir.Attribute, scale_factor_kind: ?*const mlir.Attribute, rnd: ?*const mlir.Attribute, relu: ?*const mlir.Attribute, dstTy: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    if (isPZO) |value| attributes.appendAssumeCapacity(.named(ctx, "isPZO", value));
    if (scale_factor_kind) |value| attributes.appendAssumeCapacity(.named(ctx, "scale_factor_kind", value));
    if (rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    attributes.appendAssumeCapacity(.named(ctx, "dstTy", dstTy));
    return mlir.Operation.make(ctx, "nvvm.convert.bf16x2.to.f6x2", .{
        .operands = .{ .flat = if (scale_factor) |value| &.{ src, value } else &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.bf16x2.to.f8x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_bf16x2_to_f8x2(ctx: *mlir.Context, src: *const mlir.Value, scale_factor: ?*const mlir.Value, dst_type: *const mlir.Type, isPZO: ?*const mlir.Attribute, scale_factor_kind: ?*const mlir.Attribute, rnd: ?*const mlir.Attribute, sat: ?*const mlir.Attribute, relu: ?*const mlir.Attribute, dstTy: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 6) = .empty;
    if (isPZO) |value| attributes.appendAssumeCapacity(.named(ctx, "isPZO", value));
    if (scale_factor_kind) |value| attributes.appendAssumeCapacity(.named(ctx, "scale_factor_kind", value));
    if (rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (sat) |value| attributes.appendAssumeCapacity(.named(ctx, "sat", value));
    if (relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    attributes.appendAssumeCapacity(.named(ctx, "dstTy", dstTy));
    return mlir.Operation.make(ctx, "nvvm.convert.bf16x2.to.f8x2", .{
        .operands = .{ .flat = if (scale_factor) |value| &.{ src, value } else &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.bf16x2.to.s2f6x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_bf16x2_to_s2f6x2(ctx: *mlir.Context, src: *const mlir.Value, scaleFactor: ?*const mlir.Value, dst_type: *const mlir.Type, relu: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    return mlir.Operation.make(ctx, "nvvm.convert.bf16x2.to.s2f6x2", .{
        .operands = .{ .flat = if (scaleFactor) |value| &.{ src, value } else &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.f16x2.to.f4x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_f16x2_to_f4x2(ctx: *mlir.Context, src: *const mlir.Value, scale_factor: ?*const mlir.Value, dst_type: *const mlir.Type, isPZO: ?*const mlir.Attribute, scale_factor_kind: ?*const mlir.Attribute, rnd: ?*const mlir.Attribute, relu: ?*const mlir.Attribute, dstTy: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    if (isPZO) |value| attributes.appendAssumeCapacity(.named(ctx, "isPZO", value));
    if (scale_factor_kind) |value| attributes.appendAssumeCapacity(.named(ctx, "scale_factor_kind", value));
    if (rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    attributes.appendAssumeCapacity(.named(ctx, "dstTy", dstTy));
    return mlir.Operation.make(ctx, "nvvm.convert.f16x2.to.f4x2", .{
        .operands = .{ .flat = if (scale_factor) |value| &.{ src, value } else &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.f16x2.to.f6x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_f16x2_to_f6x2(ctx: *mlir.Context, src: *const mlir.Value, scale_factor: ?*const mlir.Value, dst_type: *const mlir.Type, isPZO: ?*const mlir.Attribute, scale_factor_kind: ?*const mlir.Attribute, rnd: ?*const mlir.Attribute, relu: ?*const mlir.Attribute, dstTy: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    if (isPZO) |value| attributes.appendAssumeCapacity(.named(ctx, "isPZO", value));
    if (scale_factor_kind) |value| attributes.appendAssumeCapacity(.named(ctx, "scale_factor_kind", value));
    if (rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    attributes.appendAssumeCapacity(.named(ctx, "dstTy", dstTy));
    return mlir.Operation.make(ctx, "nvvm.convert.f16x2.to.f6x2", .{
        .operands = .{ .flat = if (scale_factor) |value| &.{ src, value } else &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.f16x2.to.f8x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_f16x2_to_f8x2(ctx: *mlir.Context, a: *const mlir.Value, scale_factor: ?*const mlir.Value, dst_type: *const mlir.Type, isPZO: ?*const mlir.Attribute, scale_factor_kind: ?*const mlir.Attribute, rnd: ?*const mlir.Attribute, sat: ?*const mlir.Attribute, relu: ?*const mlir.Attribute, dstTy: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 6) = .empty;
    if (isPZO) |value| attributes.appendAssumeCapacity(.named(ctx, "isPZO", value));
    if (scale_factor_kind) |value| attributes.appendAssumeCapacity(.named(ctx, "scale_factor_kind", value));
    if (rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (sat) |value| attributes.appendAssumeCapacity(.named(ctx, "sat", value));
    if (relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    attributes.appendAssumeCapacity(.named(ctx, "dstTy", dstTy));
    return mlir.Operation.make(ctx, "nvvm.convert.f16x2.to.f8x2", .{
        .operands = .{ .flat = if (scale_factor) |value| &.{ a, value } else &.{a} },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.f32x2.to.bf16x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_f32x2_to_bf16x2(ctx: *mlir.Context, src_hi: *const mlir.Value, src_lo: *const mlir.Value, random_bits: ?*const mlir.Value, dst_type: *const mlir.Type, rnd: ?*const mlir.Attribute, sat: ?*const mlir.Attribute, relu: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (sat) |value| attributes.appendAssumeCapacity(.named(ctx, "sat", value));
    if (relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    return mlir.Operation.make(ctx, "nvvm.convert.f32x2.to.bf16x2", .{
        .operands = .{ .flat = if (random_bits) |value| &.{ src_hi, src_lo, value } else &.{ src_hi, src_lo } },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.f32x2.to.f16x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_f32x2_to_f16x2(ctx: *mlir.Context, src_hi: *const mlir.Value, src_lo: *const mlir.Value, random_bits: ?*const mlir.Value, dst_type: *const mlir.Type, rnd: ?*const mlir.Attribute, sat: ?*const mlir.Attribute, relu: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (sat) |value| attributes.appendAssumeCapacity(.named(ctx, "sat", value));
    if (relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    return mlir.Operation.make(ctx, "nvvm.convert.f32x2.to.f16x2", .{
        .operands = .{ .flat = if (random_bits) |value| &.{ src_hi, src_lo, value } else &.{ src_hi, src_lo } },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.f32x2.to.f4x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_f32x2_to_f4x2(ctx: *mlir.Context, a: *const mlir.Value, b: *const mlir.Value, scale_factor: ?*const mlir.Value, dst_type: *const mlir.Type, isPZO: ?*const mlir.Attribute, scale_factor_kind: ?*const mlir.Attribute, rnd: ?*const mlir.Attribute, relu: ?*const mlir.Attribute, dstType: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    if (isPZO) |value| attributes.appendAssumeCapacity(.named(ctx, "isPZO", value));
    if (scale_factor_kind) |value| attributes.appendAssumeCapacity(.named(ctx, "scale_factor_kind", value));
    if (rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    attributes.appendAssumeCapacity(.named(ctx, "dstType", dstType));
    return mlir.Operation.make(ctx, "nvvm.convert.f32x2.to.f4x2", .{
        .operands = .{ .flat = if (scale_factor) |value| &.{ a, b, value } else &.{ a, b } },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.f32x2.to.f6x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_f32x2_to_f6x2(ctx: *mlir.Context, a: *const mlir.Value, b: *const mlir.Value, scale_factor: ?*const mlir.Value, dst_type: *const mlir.Type, isPZO: ?*const mlir.Attribute, scale_factor_kind: ?*const mlir.Attribute, rnd: ?*const mlir.Attribute, relu: ?*const mlir.Attribute, dstTy: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    if (isPZO) |value| attributes.appendAssumeCapacity(.named(ctx, "isPZO", value));
    if (scale_factor_kind) |value| attributes.appendAssumeCapacity(.named(ctx, "scale_factor_kind", value));
    if (rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    attributes.appendAssumeCapacity(.named(ctx, "dstTy", dstTy));
    return mlir.Operation.make(ctx, "nvvm.convert.f32x2.to.f6x2", .{
        .operands = .{ .flat = if (scale_factor) |value| &.{ a, b, value } else &.{ a, b } },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.f32x2.to.f8x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_f32x2_to_f8x2(ctx: *mlir.Context, a: *const mlir.Value, b: *const mlir.Value, scale_factor: ?*const mlir.Value, dst_type: *const mlir.Type, isPZO: ?*const mlir.Attribute, scale_factor_kind: ?*const mlir.Attribute, rnd: ?*const mlir.Attribute, sat: ?*const mlir.Attribute, relu: ?*const mlir.Attribute, dstTy: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 6) = .empty;
    if (isPZO) |value| attributes.appendAssumeCapacity(.named(ctx, "isPZO", value));
    if (scale_factor_kind) |value| attributes.appendAssumeCapacity(.named(ctx, "scale_factor_kind", value));
    if (rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (sat) |value| attributes.appendAssumeCapacity(.named(ctx, "sat", value));
    if (relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    attributes.appendAssumeCapacity(.named(ctx, "dstTy", dstTy));
    return mlir.Operation.make(ctx, "nvvm.convert.f32x2.to.f8x2", .{
        .operands = .{ .flat = if (scale_factor) |value| &.{ a, b, value } else &.{ a, b } },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.f32x2.to.s2f6x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_f32x2_to_s2f6x2(ctx: *mlir.Context, a: *const mlir.Value, b: *const mlir.Value, scaleFactor: ?*const mlir.Value, dst_type: *const mlir.Type, relu: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    return mlir.Operation.make(ctx, "nvvm.convert.f32x2.to.s2f6x2", .{
        .operands = .{ .flat = if (scaleFactor) |value| &.{ a, b, value } else &.{ a, b } },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.f32x4.to.f4x4`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_f32x4_to_f4x4(ctx: *mlir.Context, src: *const mlir.Value, rbits: *const mlir.Value, dst_type: *const mlir.Type, relu: ?*const mlir.Attribute, dstTy: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    attributes.appendAssumeCapacity(.named(ctx, "dstTy", dstTy));
    return mlir.Operation.make(ctx, "nvvm.convert.f32x4.to.f4x4", .{
        .operands = .{ .flat = &.{ src, rbits } },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.f32x4.to.f6x4`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_f32x4_to_f6x4(ctx: *mlir.Context, src: *const mlir.Value, rbits: *const mlir.Value, dst_type: *const mlir.Type, relu: ?*const mlir.Attribute, dstTy: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    attributes.appendAssumeCapacity(.named(ctx, "dstTy", dstTy));
    return mlir.Operation.make(ctx, "nvvm.convert.f32x4.to.f6x4", .{
        .operands = .{ .flat = &.{ src, rbits } },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.f32x4.to.f8x4`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_f32x4_to_f8x4(ctx: *mlir.Context, src: *const mlir.Value, rbits: *const mlir.Value, dst_type: *const mlir.Type, relu: ?*const mlir.Attribute, dstTy: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    attributes.appendAssumeCapacity(.named(ctx, "dstTy", dstTy));
    return mlir.Operation.make(ctx, "nvvm.convert.f32x4.to.f8x4", .{
        .operands = .{ .flat = &.{ src, rbits } },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.f4x2.to.f16x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_f4x2_to_f16x2(ctx: *mlir.Context, src: *const mlir.Value, dst_type: *const mlir.Type, relu: ?*const mlir.Attribute, srcType: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    attributes.appendAssumeCapacity(.named(ctx, "srcType", srcType));
    return mlir.Operation.make(ctx, "nvvm.convert.f4x2.to.f16x2", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.f6x2.to.f16x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_f6x2_to_f16x2(ctx: *mlir.Context, src: *const mlir.Value, dst_type: *const mlir.Type, relu: ?*const mlir.Attribute, srcType: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    attributes.appendAssumeCapacity(.named(ctx, "srcType", srcType));
    return mlir.Operation.make(ctx, "nvvm.convert.f6x2.to.f16x2", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.f8x2.to.bf16x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_f8x2_to_bf16x2(ctx: *mlir.Context, src: *const mlir.Value, scaleFactor: ?*const mlir.Value, dst_type: *const mlir.Type, sat: ?*const mlir.Attribute, srcType: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (sat) |value| attributes.appendAssumeCapacity(.named(ctx, "sat", value));
    attributes.appendAssumeCapacity(.named(ctx, "srcType", srcType));
    return mlir.Operation.make(ctx, "nvvm.convert.f8x2.to.bf16x2", .{
        .operands = .{ .flat = if (scaleFactor) |value| &.{ src, value } else &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.f8x2.to.f16x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_f8x2_to_f16x2(ctx: *mlir.Context, src: *const mlir.Value, dst_type: *const mlir.Type, relu: ?*const mlir.Attribute, srcType: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    attributes.appendAssumeCapacity(.named(ctx, "srcType", srcType));
    return mlir.Operation.make(ctx, "nvvm.convert.f8x2.to.f16x2", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.float.to.integer`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_float_to_integer(ctx: *mlir.Context, src: *const mlir.Value, dst_type: *const mlir.Type, rnd: ?*const mlir.Attribute, sat: ?*const mlir.Attribute, ftz: ?*const mlir.Attribute, is_signed: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    if (rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (sat) |value| attributes.appendAssumeCapacity(.named(ctx, "sat", value));
    if (ftz) |value| attributes.appendAssumeCapacity(.named(ctx, "ftz", value));
    if (is_signed) |value| attributes.appendAssumeCapacity(.named(ctx, "is_signed", value));
    return mlir.Operation.make(ctx, "nvvm.convert.float.to.integer", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.float.to.tf32`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_float_to_tf32(ctx: *mlir.Context, src: *const mlir.Value, res_type: *const mlir.Type, rnd: ?*const mlir.Attribute, sat: ?*const mlir.Attribute, relu: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (sat) |value| attributes.appendAssumeCapacity(.named(ctx, "sat", value));
    if (relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    return mlir.Operation.make(ctx, "nvvm.convert.float.to.tf32", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.s2f6x2.to.bf16x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_s2f6x2_to_bf16x2(ctx: *mlir.Context, src: *const mlir.Value, scaleFactor: ?*const mlir.Value, dst_type: *const mlir.Type, sat: ?*const mlir.Attribute, relu: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (sat) |value| attributes.appendAssumeCapacity(.named(ctx, "sat", value));
    if (relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    return mlir.Operation.make(ctx, "nvvm.convert.s2f6x2.to.bf16x2", .{
        .operands = .{ .flat = if (scaleFactor) |value| &.{ src, value } else &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cos`. Result types are explicit; attributes use mlir.Attribute.
pub fn cos(ctx: *mlir.Context, src: *const mlir.Value, res_type: *const mlir.Type, ftz: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (ftz) |value| attributes.appendAssumeCapacity(.named(ctx, "ftz", value));
    return mlir.Operation.make(ctx, "nvvm.cos", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cp.async.bulk.commit.group`. Result types are explicit; attributes use mlir.Attribute.
pub fn cp_async_bulk_commit_group(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.cp.async.bulk.commit.group", .{
        .location = location,
    });
}

/// `nvvm.cp.async.bulk.global.shared.cta`. Result types are explicit; attributes use mlir.Attribute.
pub fn cp_async_bulk_global_shared_cta(ctx: *mlir.Context, dstMem: *const mlir.Value, srcMem: *const mlir.Value, size: *const mlir.Value, l2CacheHint: ?*const mlir.Value, byteMask: ?*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.cp.async.bulk.global.shared.cta", .{
        .operands = .{ .variadic = &.{
            &.{dstMem},
            &.{srcMem},
            &.{size},
            if (l2CacheHint) |value| &.{value} else &.{},
            if (byteMask) |value| &.{value} else &.{},
        } },
        .location = location,
    });
}

/// `nvvm.cp.async.bulk.prefetch`. Result types are explicit; attributes use mlir.Attribute.
pub fn cp_async_bulk_prefetch(ctx: *mlir.Context, srcMem: *const mlir.Value, size: *const mlir.Value, l2CacheHint: ?*const mlir.Value, evict: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (evict) |value| attributes.appendAssumeCapacity(.named(ctx, "evict", value));
    return mlir.Operation.make(ctx, "nvvm.cp.async.bulk.prefetch", .{
        .operands = .{ .flat = if (l2CacheHint) |value| &.{ srcMem, size, value } else &.{ srcMem, size } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cp.async.bulk.shared.cluster.global`. Result types are explicit; attributes use mlir.Attribute.
pub fn cp_async_bulk_shared_cluster_global(ctx: *mlir.Context, dstMem: *const mlir.Value, srcMem: *const mlir.Value, mbar: *const mlir.Value, size: *const mlir.Value, multicastMask: ?*const mlir.Value, l2CacheHint: ?*const mlir.Value, validatePattern: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (validatePattern) |value| attributes.appendAssumeCapacity(.named(ctx, "validatePattern", value));
    return mlir.Operation.make(ctx, "nvvm.cp.async.bulk.shared.cluster.global", .{
        .operands = .{ .variadic = &.{
            &.{dstMem},
            &.{srcMem},
            &.{mbar},
            &.{size},
            if (multicastMask) |value| &.{value} else &.{},
            if (l2CacheHint) |value| &.{value} else &.{},
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cp.async.bulk.shared.cluster.shared.cta`. Result types are explicit; attributes use mlir.Attribute.
pub fn cp_async_bulk_shared_cluster_shared_cta(ctx: *mlir.Context, dstMem: *const mlir.Value, srcMem: *const mlir.Value, mbar: *const mlir.Value, size: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.cp.async.bulk.shared.cluster.shared.cta", .{
        .operands = .{ .flat = &.{ dstMem, srcMem, mbar, size } },
        .location = location,
    });
}

/// `nvvm.cp.async.bulk.tensor.global.shared.cta`. Result types are explicit; attributes use mlir.Attribute.
pub fn cp_async_bulk_tensor_global_shared_cta(ctx: *mlir.Context, tmaDescriptor: *const mlir.Value, srcMem: *const mlir.Value, coordinates: []const *const mlir.Value, l2CacheHint: ?*const mlir.Value, predicate: ?*const mlir.Value, mode: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (mode) |value| attributes.appendAssumeCapacity(.named(ctx, "mode", value));
    return mlir.Operation.make(ctx, "nvvm.cp.async.bulk.tensor.global.shared.cta", .{
        .operands = .{ .variadic = &.{
            &.{tmaDescriptor},
            &.{srcMem},
            coordinates,
            if (l2CacheHint) |value| &.{value} else &.{},
            if (predicate) |value| &.{value} else &.{},
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cp.async.bulk.tensor.global.shared.cta.override`. Result types are explicit; attributes use mlir.Attribute.
pub fn cp_async_bulk_tensor_global_shared_cta_override(ctx: *mlir.Context, tmaDesc: *const mlir.Value, srcMem: *const mlir.Value, overrideAdrr: *const mlir.Value, coordinates: []const *const mlir.Value, tensorSize: []const *const mlir.Value, lowerStride: []const *const mlir.Value, upperStride: ?*const mlir.Value, l2CacheHint: ?*const mlir.Value, mode: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (mode) |value| attributes.appendAssumeCapacity(.named(ctx, "mode", value));
    return mlir.Operation.make(ctx, "nvvm.cp.async.bulk.tensor.global.shared.cta.override", .{
        .operands = .{ .variadic = &.{
            &.{tmaDesc},
            &.{srcMem},
            &.{overrideAdrr},
            coordinates,
            tensorSize,
            lowerStride,
            if (upperStride) |value| &.{value} else &.{},
            if (l2CacheHint) |value| &.{value} else &.{},
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cp.async.bulk.tensor.prefetch`. Result types are explicit; attributes use mlir.Attribute.
pub fn cp_async_bulk_tensor_prefetch(ctx: *mlir.Context, tmaDescriptor: *const mlir.Value, coordinates: []const *const mlir.Value, im2colOffsets: []const *const mlir.Value, l2CacheHint: ?*const mlir.Value, mode: ?*const mlir.Attribute, evict: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (mode) |value| attributes.appendAssumeCapacity(.named(ctx, "mode", value));
    if (evict) |value| attributes.appendAssumeCapacity(.named(ctx, "evict", value));
    return mlir.Operation.make(ctx, "nvvm.cp.async.bulk.tensor.prefetch", .{
        .operands = .{ .variadic = &.{
            &.{tmaDescriptor},
            coordinates,
            im2colOffsets,
            if (l2CacheHint) |value| &.{value} else &.{},
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cp.async.bulk.tensor.prefetch.override`. Result types are explicit; attributes use mlir.Attribute.
pub fn cp_async_bulk_tensor_prefetch_override(ctx: *mlir.Context, tmaDesc: *const mlir.Value, overrideAdrr: *const mlir.Value, coordinates: []const *const mlir.Value, tensorSize: []const *const mlir.Value, lowerStride: []const *const mlir.Value, upperStride: ?*const mlir.Value, im2colOffsets: []const *const mlir.Value, l2CacheHint: ?*const mlir.Value, evict: ?*const mlir.Attribute, mode: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (evict) |value| attributes.appendAssumeCapacity(.named(ctx, "evict", value));
    if (mode) |value| attributes.appendAssumeCapacity(.named(ctx, "mode", value));
    return mlir.Operation.make(ctx, "nvvm.cp.async.bulk.tensor.prefetch.override", .{
        .operands = .{ .variadic = &.{
            &.{tmaDesc},
            &.{overrideAdrr},
            coordinates,
            tensorSize,
            lowerStride,
            if (upperStride) |value| &.{value} else &.{},
            im2colOffsets,
            if (l2CacheHint) |value| &.{value} else &.{},
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cp.async.bulk.tensor.reduce`. Result types are explicit; attributes use mlir.Attribute.
pub fn cp_async_bulk_tensor_reduce(ctx: *mlir.Context, tmaDescriptor: *const mlir.Value, srcMem: *const mlir.Value, coordinates: []const *const mlir.Value, l2CacheHint: ?*const mlir.Value, redKind: *const mlir.Attribute, mode: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "redKind", redKind));
    if (mode) |value| attributes.appendAssumeCapacity(.named(ctx, "mode", value));
    return mlir.Operation.make(ctx, "nvvm.cp.async.bulk.tensor.reduce", .{
        .operands = .{ .variadic = &.{
            &.{tmaDescriptor},
            &.{srcMem},
            coordinates,
            if (l2CacheHint) |value| &.{value} else &.{},
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cp.async.bulk.tensor.reduce.override`. Result types are explicit; attributes use mlir.Attribute.
pub fn cp_async_bulk_tensor_reduce_override(ctx: *mlir.Context, tmaDesc: *const mlir.Value, srcMem: *const mlir.Value, overrideAdrr: *const mlir.Value, coordinates: []const *const mlir.Value, tensorSize: []const *const mlir.Value, lowerStride: []const *const mlir.Value, upperStride: ?*const mlir.Value, l2CacheHint: ?*const mlir.Value, redKind: *const mlir.Attribute, mode: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "redKind", redKind));
    if (mode) |value| attributes.appendAssumeCapacity(.named(ctx, "mode", value));
    return mlir.Operation.make(ctx, "nvvm.cp.async.bulk.tensor.reduce.override", .{
        .operands = .{ .variadic = &.{
            &.{tmaDesc},
            &.{srcMem},
            &.{overrideAdrr},
            coordinates,
            tensorSize,
            lowerStride,
            if (upperStride) |value| &.{value} else &.{},
            if (l2CacheHint) |value| &.{value} else &.{},
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cp.async.bulk.tensor.shared.cluster.global`. Result types are explicit; attributes use mlir.Attribute.
pub fn cp_async_bulk_tensor_shared_cluster_global(ctx: *mlir.Context, dstMem: *const mlir.Value, tmaDescriptor: *const mlir.Value, coordinates: []const *const mlir.Value, mbar: *const mlir.Value, im2colOffsets: []const *const mlir.Value, multicastMask: ?*const mlir.Value, l2CacheHint: ?*const mlir.Value, predicate: ?*const mlir.Value, validatePattern: ?*const mlir.Attribute, mode: ?*const mlir.Attribute, isCTAOnly: ?*const mlir.Attribute, group: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    if (validatePattern) |value| attributes.appendAssumeCapacity(.named(ctx, "validatePattern", value));
    if (mode) |value| attributes.appendAssumeCapacity(.named(ctx, "mode", value));
    if (isCTAOnly) |value| attributes.appendAssumeCapacity(.named(ctx, "isCTAOnly", value));
    if (group) |value| attributes.appendAssumeCapacity(.named(ctx, "group", value));
    return mlir.Operation.make(ctx, "nvvm.cp.async.bulk.tensor.shared.cluster.global", .{
        .operands = .{ .variadic = &.{
            &.{dstMem},
            &.{tmaDescriptor},
            coordinates,
            &.{mbar},
            im2colOffsets,
            if (multicastMask) |value| &.{value} else &.{},
            if (l2CacheHint) |value| &.{value} else &.{},
            if (predicate) |value| &.{value} else &.{},
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cp.async.bulk.tensor.shared.cluster.global.override`. Result types are explicit; attributes use mlir.Attribute.
pub fn cp_async_bulk_tensor_shared_cluster_global_override(ctx: *mlir.Context, dstMem: *const mlir.Value, tmaDesc: *const mlir.Value, mbar: *const mlir.Value, overrideAddr: *const mlir.Value, coordinates: []const *const mlir.Value, im2colArgs: []const *const mlir.Value, tensorSize: []const *const mlir.Value, lowerStride: []const *const mlir.Value, upperStride: ?*const mlir.Value, multicastMask: ?*const mlir.Value, l2CacheHint: ?*const mlir.Value, group: ?*const mlir.Attribute, validatePattern: ?*const mlir.Attribute, mode: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (group) |value| attributes.appendAssumeCapacity(.named(ctx, "group", value));
    if (validatePattern) |value| attributes.appendAssumeCapacity(.named(ctx, "validatePattern", value));
    if (mode) |value| attributes.appendAssumeCapacity(.named(ctx, "mode", value));
    return mlir.Operation.make(ctx, "nvvm.cp.async.bulk.tensor.shared.cluster.global.override", .{
        .operands = .{ .variadic = &.{
            &.{dstMem},
            &.{tmaDesc},
            &.{mbar},
            &.{overrideAddr},
            coordinates,
            im2colArgs,
            tensorSize,
            lowerStride,
            if (upperStride) |value| &.{value} else &.{},
            if (multicastMask) |value| &.{value} else &.{},
            if (l2CacheHint) |value| &.{value} else &.{},
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cp.async.bulk.wait_group`. Result types are explicit; attributes use mlir.Attribute.
pub fn cp_async_bulk_wait_group(ctx: *mlir.Context, group: *const mlir.Attribute, read: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "group", group));
    if (read) |value| attributes.appendAssumeCapacity(.named(ctx, "read", value));
    return mlir.Operation.make(ctx, "nvvm.cp.async.bulk.wait_group", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cp.async.commit.group`. Result types are explicit; attributes use mlir.Attribute.
pub fn cp_async_commit_group(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.cp.async.commit.group", .{
        .location = location,
    });
}

/// `nvvm.cp.async.mbarrier.arrive`. Result types are explicit; attributes use mlir.Attribute.
pub fn cp_async_mbarrier_arrive(ctx: *mlir.Context, addr: *const mlir.Value, noinc: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (noinc) |value| attributes.appendAssumeCapacity(.named(ctx, "noinc", value));
    return mlir.Operation.make(ctx, "nvvm.cp.async.mbarrier.arrive", .{
        .operands = .{ .flat = &.{addr} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cp.async.shared.global`. Result types are explicit; attributes use mlir.Attribute.
pub fn cp_async_shared_global(ctx: *mlir.Context, dst: *const mlir.Value, src: *const mlir.Value, cpSize: ?*const mlir.Value, size: *const mlir.Attribute, modifier: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.cp.async.shared.global", .{
        .operands = .{ .flat = if (cpSize) |value| &.{ dst, src, value } else &.{ dst, src } },
        .attributes = &.{
            .named(ctx, "size", size),
            .named(ctx, "modifier", modifier),
        },
        .location = location,
    });
}

/// `nvvm.cp.async.wait.group`. Result types are explicit; attributes use mlir.Attribute.
pub fn cp_async_wait_group(ctx: *mlir.Context, n: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.cp.async.wait.group", .{
        .attributes = &.{
            .named(ctx, "n", n),
        },
        .location = location,
    });
}

/// `nvvm.cvt.packfloat`. Result types are explicit; attributes use mlir.Attribute.
pub fn cvt_packfloat(ctx: *mlir.Context, srcA: *const mlir.Value, srcC: *const mlir.Value, res_type: *const mlir.Type, from: *const mlir.Attribute, to: *const mlir.Attribute, rnd: ?*const mlir.Attribute, sat: ?*const mlir.Attribute, relu: ?*const mlir.Attribute, extractHi: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 6) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "from", from));
    attributes.appendAssumeCapacity(.named(ctx, "to", to));
    if (rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (sat) |value| attributes.appendAssumeCapacity(.named(ctx, "sat", value));
    if (relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    if (extractHi) |value| attributes.appendAssumeCapacity(.named(ctx, "extractHi", value));
    return mlir.Operation.make(ctx, "nvvm.cvt.packfloat", .{
        .operands = .{ .flat = &.{ srcA, srcC } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cvt.packfloat.f32`. Result types are explicit; attributes use mlir.Attribute.
pub fn cvt_packfloat_f32(ctx: *mlir.Context, srcA: *const mlir.Value, srcB: *const mlir.Value, srcC: *const mlir.Value, res_type: *const mlir.Type, to: *const mlir.Attribute, rnd: ?*const mlir.Attribute, sat: ?*const mlir.Attribute, relu: ?*const mlir.Attribute, extractHi: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "to", to));
    if (rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (sat) |value| attributes.appendAssumeCapacity(.named(ctx, "sat", value));
    if (relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    if (extractHi) |value| attributes.appendAssumeCapacity(.named(ctx, "extractHi", value));
    return mlir.Operation.make(ctx, "nvvm.cvt.packfloat.f32", .{
        .operands = .{ .flat = &.{ srcA, srcB, srcC } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cvt.to.f4x8.packed`. Result types are explicit; attributes use mlir.Attribute.
pub fn cvt_to_f4x8_packed(ctx: *mlir.Context, srcs: []const *const mlir.Value, dst_type: *const mlir.Type, rnd: ?*const mlir.Attribute, srcType: *const mlir.Attribute, dstType: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    attributes.appendAssumeCapacity(.named(ctx, "srcType", srcType));
    attributes.appendAssumeCapacity(.named(ctx, "dstType", dstType));
    return mlir.Operation.make(ctx, "nvvm.cvt.to.f4x8.packed", .{
        .operands = .{ .flat = srcs },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.dot.accumulate.2way`. Result types are explicit; attributes use mlir.Attribute.
pub fn dot_accumulate_2way(ctx: *mlir.Context, a: *const mlir.Value, b: *const mlir.Value, c_: *const mlir.Value, res_type: *const mlir.Type, a_type: *const mlir.Attribute, b_type: *const mlir.Attribute, b_hi: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.dot.accumulate.2way", .{
        .operands = .{ .flat = &.{ a, b, c_ } },
        .results = .{ .flat = &.{res_type} },
        .attributes = &.{
            .named(ctx, "a_type", a_type),
            .named(ctx, "b_type", b_type),
            .named(ctx, "b_hi", b_hi),
        },
        .location = location,
    });
}

/// `nvvm.dot.accumulate.4way`. Result types are explicit; attributes use mlir.Attribute.
pub fn dot_accumulate_4way(ctx: *mlir.Context, a: *const mlir.Value, b: *const mlir.Value, c_: *const mlir.Value, res_type: *const mlir.Type, a_type: *const mlir.Attribute, b_type: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.dot.accumulate.4way", .{
        .operands = .{ .flat = &.{ a, b, c_ } },
        .results = .{ .flat = &.{res_type} },
        .attributes = &.{
            .named(ctx, "a_type", a_type),
            .named(ctx, "b_type", b_type),
        },
        .location = location,
    });
}

/// `nvvm.elect.sync`. Result types are explicit; attributes use mlir.Attribute.
pub fn elect_sync(ctx: *mlir.Context, membermask: ?*const mlir.Value, pred_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.elect.sync", .{
        .operands = .{ .flat = if (membermask) |value| &.{value} else &.{} },
        .results = .{ .flat = &.{pred_type} },
        .location = location,
    });
}

/// `nvvm.ex2`. Result types are explicit; attributes use mlir.Attribute.
pub fn ex2(ctx: *mlir.Context, src: *const mlir.Value, res_type: *const mlir.Type, ftz: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (ftz) |value| attributes.appendAssumeCapacity(.named(ctx, "ftz", value));
    return mlir.Operation.make(ctx, "nvvm.ex2", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.exit`. Result types are explicit; attributes use mlir.Attribute.
pub fn exit(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.exit", .{
        .location = location,
    });
}

/// `nvvm.fabs`. Result types are explicit; attributes use mlir.Attribute.
pub fn fabs(ctx: *mlir.Context, value_: *const mlir.Value, res_type: *const mlir.Type, ftz: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (ftz) |value| attributes.appendAssumeCapacity(.named(ctx, "ftz", value));
    return mlir.Operation.make(ctx, "nvvm.fabs", .{
        .operands = .{ .flat = &.{value_} },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.fence.mbarrier.init`. Result types are explicit; attributes use mlir.Attribute.
pub fn fence_mbarrier_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.fence.mbarrier.init", .{
        .location = location,
    });
}

/// `nvvm.fence.proxy`. Result types are explicit; attributes use mlir.Attribute.
pub fn fence_proxy(ctx: *mlir.Context, kind: *const mlir.Attribute, space: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (space) |value| attributes.appendAssumeCapacity(.named(ctx, "space", value));
    return mlir.Operation.make(ctx, "nvvm.fence.proxy", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.fence.proxy.acquire`. Result types are explicit; attributes use mlir.Attribute.
pub fn fence_proxy_acquire(ctx: *mlir.Context, addr: *const mlir.Value, size: *const mlir.Value, scope: *const mlir.Attribute, fromProxy: ?*const mlir.Attribute, toProxy: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "scope", scope));
    if (fromProxy) |value| attributes.appendAssumeCapacity(.named(ctx, "fromProxy", value));
    if (toProxy) |value| attributes.appendAssumeCapacity(.named(ctx, "toProxy", value));
    return mlir.Operation.make(ctx, "nvvm.fence.proxy.acquire", .{
        .operands = .{ .flat = &.{ addr, size } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.fence.proxy.release`. Result types are explicit; attributes use mlir.Attribute.
pub fn fence_proxy_release(ctx: *mlir.Context, scope: *const mlir.Attribute, fromProxy: ?*const mlir.Attribute, toProxy: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "scope", scope));
    if (fromProxy) |value| attributes.appendAssumeCapacity(.named(ctx, "fromProxy", value));
    if (toProxy) |value| attributes.appendAssumeCapacity(.named(ctx, "toProxy", value));
    return mlir.Operation.make(ctx, "nvvm.fence.proxy.release", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.fence.proxy.sync_restrict`. Result types are explicit; attributes use mlir.Attribute.
pub fn fence_proxy_sync_restrict(ctx: *mlir.Context, order: *const mlir.Attribute, fromProxy: ?*const mlir.Attribute, toProxy: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "order", order));
    if (fromProxy) |value| attributes.appendAssumeCapacity(.named(ctx, "fromProxy", value));
    if (toProxy) |value| attributes.appendAssumeCapacity(.named(ctx, "toProxy", value));
    return mlir.Operation.make(ctx, "nvvm.fence.proxy.sync_restrict", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.fence.sc.cluster`. Result types are explicit; attributes use mlir.Attribute.
pub fn fence_sc_cluster(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.fence.sc.cluster", .{
        .location = location,
    });
}

/// `nvvm.fence.sync_restrict`. Result types are explicit; attributes use mlir.Attribute.
pub fn fence_sync_restrict(ctx: *mlir.Context, order: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.fence.sync_restrict", .{
        .attributes = &.{
            .named(ctx, "order", order),
        },
        .location = location,
    });
}

/// `nvvm.fma`. Result types are explicit; attributes use mlir.Attribute.
pub fn fma(ctx: *mlir.Context, a: *const mlir.Value, b: *const mlir.Value, c_: *const mlir.Value, res_type: *const mlir.Type, rnd: *const mlir.Attribute, sat: ?*const mlir.Attribute, ftz: ?*const mlir.Attribute, relu: ?*const mlir.Attribute, oob: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "rnd", rnd));
    if (sat) |value| attributes.appendAssumeCapacity(.named(ctx, "sat", value));
    if (ftz) |value| attributes.appendAssumeCapacity(.named(ctx, "ftz", value));
    if (relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    if (oob) |value| attributes.appendAssumeCapacity(.named(ctx, "oob", value));
    return mlir.Operation.make(ctx, "nvvm.fma", .{
        .operands = .{ .flat = &.{ a, b, c_ } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.fma.packed.f32x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn fma_packed_f32x2(ctx: *mlir.Context, srcA: *const mlir.Value, srcB: *const mlir.Value, srcC: *const mlir.Value, res_type: *const mlir.Type, rnd: ?*const mlir.Attribute, ftz: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (ftz) |value| attributes.appendAssumeCapacity(.named(ctx, "ftz", value));
    return mlir.Operation.make(ctx, "nvvm.fma.packed.f32x2", .{
        .operands = .{ .flat = &.{ srcA, srcB, srcC } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.fma.packed.f32x2.bf16x2.f32x2.f32x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn fma_packed_f32x2_bf16x2_f32x2_f32x2(ctx: *mlir.Context, srcA: *const mlir.Value, srcB: *const mlir.Value, srcC: *const mlir.Value, res_type: *const mlir.Type, rnd: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    return mlir.Operation.make(ctx, "nvvm.fma.packed.f32x2.bf16x2.f32x2.f32x2", .{
        .operands = .{ .flat = &.{ srcA, srcB, srcC } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.fma.packed.f32x2.f16x2.f32x2.f32x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn fma_packed_f32x2_f16x2_f32x2_f32x2(ctx: *mlir.Context, srcA: *const mlir.Value, srcB: *const mlir.Value, srcC: *const mlir.Value, res_type: *const mlir.Type, rnd: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    return mlir.Operation.make(ctx, "nvvm.fma.packed.f32x2.f16x2.f32x2.f32x2", .{
        .operands = .{ .flat = &.{ srcA, srcB, srcC } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.fmax`. Result types are explicit; attributes use mlir.Attribute.
pub fn fmax(ctx: *mlir.Context, a: *const mlir.Value, b: *const mlir.Value, c_: ?*const mlir.Value, res_type: *const mlir.Type, ftz: ?*const mlir.Attribute, nan: ?*const mlir.Attribute, abs: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (ftz) |value| attributes.appendAssumeCapacity(.named(ctx, "ftz", value));
    if (nan) |value| attributes.appendAssumeCapacity(.named(ctx, "nan", value));
    if (abs) |value| attributes.appendAssumeCapacity(.named(ctx, "abs", value));
    return mlir.Operation.make(ctx, "nvvm.fmax", .{
        .operands = .{ .flat = if (c_) |value| &.{ a, b, value } else &.{ a, b } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.fmin`. Result types are explicit; attributes use mlir.Attribute.
pub fn fmin(ctx: *mlir.Context, a: *const mlir.Value, b: *const mlir.Value, c_: ?*const mlir.Value, res_type: *const mlir.Type, ftz: ?*const mlir.Attribute, nan: ?*const mlir.Attribute, abs: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (ftz) |value| attributes.appendAssumeCapacity(.named(ctx, "ftz", value));
    if (nan) |value| attributes.appendAssumeCapacity(.named(ctx, "nan", value));
    if (abs) |value| attributes.appendAssumeCapacity(.named(ctx, "abs", value));
    return mlir.Operation.make(ctx, "nvvm.fmin", .{
        .operands = .{ .flat = if (c_) |value| &.{ a, b, value } else &.{ a, b } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.griddepcontrol`. Result types are explicit; attributes use mlir.Attribute.
pub fn griddepcontrol(ctx: *mlir.Context, kind: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.griddepcontrol", .{
        .attributes = &.{
            .named(ctx, "kind", kind),
        },
        .location = location,
    });
}

/// `nvvm.inline_ptx`. Result types are explicit; attributes use mlir.Attribute.
pub fn inline_ptx(ctx: *mlir.Context, readOnlyArgs: []const *const mlir.Value, readWriteArgs: []const *const mlir.Value, predicate: ?*const mlir.Value, writeOnlyArgs_type: ?*const mlir.Type, ptxCode: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var results: stdx.BoundedArray(*const mlir.Type, 256) = .empty;
    if (writeOnlyArgs_type) |value| results.appendAssumeCapacity(value);
    return mlir.Operation.make(ctx, "nvvm.inline_ptx", .{
        .operands = .{ .variadic = &.{
            readOnlyArgs,
            readWriteArgs,
            if (predicate) |value| &.{value} else &.{},
        } },
        .results = .{ .flat = results.constSlice() },
        .attributes = &.{
            .named(ctx, "ptxCode", ptxCode),
        },
        .location = location,
    });
}

/// `nvvm.ldmatrix`. Result types are explicit; attributes use mlir.Attribute.
pub fn ldmatrix(ctx: *mlir.Context, ptr: *const mlir.Value, res_type: *const mlir.Type, num: *const mlir.Attribute, layout: *const mlir.Attribute, shape: ?*const mlir.Attribute, srcFormat: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "num", num));
    attributes.appendAssumeCapacity(.named(ctx, "layout", layout));
    if (shape) |value| attributes.appendAssumeCapacity(.named(ctx, "shape", value));
    if (srcFormat) |value| attributes.appendAssumeCapacity(.named(ctx, "srcFormat", value));
    return mlir.Operation.make(ctx, "nvvm.ldmatrix", .{
        .operands = .{ .flat = &.{ptr} },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.load.ext`. Result types are explicit; attributes use mlir.Attribute.
pub fn load_ext(ctx: *mlir.Context, addr: *const mlir.Value, l2CacheHint: ?*const mlir.Value, res_type: *const mlir.Type, order: ?*const mlir.Attribute, scope: ?*const mlir.Attribute, prefetch_: ?*const mlir.Attribute, evict: ?*const mlir.Attribute, cacheModifier: ?*const mlir.Attribute, sharedSpace: ?*const mlir.Attribute, unified: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 7) = .empty;
    if (order) |value| attributes.appendAssumeCapacity(.named(ctx, "order", value));
    if (scope) |value| attributes.appendAssumeCapacity(.named(ctx, "scope", value));
    if (prefetch_) |value| attributes.appendAssumeCapacity(.named(ctx, "prefetch", value));
    if (evict) |value| attributes.appendAssumeCapacity(.named(ctx, "evict", value));
    if (cacheModifier) |value| attributes.appendAssumeCapacity(.named(ctx, "cacheModifier", value));
    if (sharedSpace) |value| attributes.appendAssumeCapacity(.named(ctx, "sharedSpace", value));
    if (unified) |value| attributes.appendAssumeCapacity(.named(ctx, "unified", value));
    return mlir.Operation.make(ctx, "nvvm.load.ext", .{
        .operands = .{ .flat = if (l2CacheHint) |value| &.{ addr, value } else &.{addr} },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.log2`. Result types are explicit; attributes use mlir.Attribute.
pub fn log2(ctx: *mlir.Context, src: *const mlir.Value, res_type: *const mlir.Type, ftz: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (ftz) |value| attributes.appendAssumeCapacity(.named(ctx, "ftz", value));
    return mlir.Operation.make(ctx, "nvvm.log2", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mapa`. Result types are explicit; attributes use mlir.Attribute.
pub fn mapa(ctx: *mlir.Context, a: *const mlir.Value, b: *const mlir.Value, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.mapa", .{
        .operands = .{ .flat = &.{ a, b } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.match.sync`. Result types are explicit; attributes use mlir.Attribute.
pub fn match_sync(ctx: *mlir.Context, thread_mask: *const mlir.Value, val: *const mlir.Value, res_type: *const mlir.Type, kind: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.match.sync", .{
        .operands = .{ .flat = &.{ thread_mask, val } },
        .results = .{ .flat = &.{res_type} },
        .attributes = &.{
            .named(ctx, "kind", kind),
        },
        .location = location,
    });
}

/// `nvvm.mbarrier.arrive`. Result types are explicit; attributes use mlir.Attribute.
pub fn mbarrier_arrive(ctx: *mlir.Context, addr: *const mlir.Value, count: ?*const mlir.Value, multicastMask: ?*const mlir.Value, res_type: ?*const mlir.Type, scope: ?*const mlir.Attribute, relaxed: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var results: stdx.BoundedArray(*const mlir.Type, 256) = .empty;
    if (res_type) |value| results.appendAssumeCapacity(value);
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (scope) |value| attributes.appendAssumeCapacity(.named(ctx, "scope", value));
    if (relaxed) |value| attributes.appendAssumeCapacity(.named(ctx, "relaxed", value));
    return mlir.Operation.make(ctx, "nvvm.mbarrier.arrive", .{
        .operands = .{ .variadic = &.{
            &.{addr},
            if (count) |value| &.{value} else &.{},
            if (multicastMask) |value| &.{value} else &.{},
        } },
        .results = .{ .flat = results.constSlice() },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mbarrier.arrive.expect_tx`. Result types are explicit; attributes use mlir.Attribute.
pub fn mbarrier_arrive_expect_tx(ctx: *mlir.Context, addr: *const mlir.Value, txcount: *const mlir.Value, multicastMask: ?*const mlir.Value, predicate: ?*const mlir.Value, res_type: ?*const mlir.Type, scope: ?*const mlir.Attribute, relaxed: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var results: stdx.BoundedArray(*const mlir.Type, 256) = .empty;
    if (res_type) |value| results.appendAssumeCapacity(value);
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (scope) |value| attributes.appendAssumeCapacity(.named(ctx, "scope", value));
    if (relaxed) |value| attributes.appendAssumeCapacity(.named(ctx, "relaxed", value));
    return mlir.Operation.make(ctx, "nvvm.mbarrier.arrive.expect_tx", .{
        .operands = .{ .variadic = &.{
            &.{addr},
            &.{txcount},
            if (multicastMask) |value| &.{value} else &.{},
            if (predicate) |value| &.{value} else &.{},
        } },
        .results = .{ .flat = results.constSlice() },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mbarrier.arrive.nocomplete`. Result types are explicit; attributes use mlir.Attribute.
pub fn mbarrier_arrive_nocomplete(ctx: *mlir.Context, addr: *const mlir.Value, count: *const mlir.Value, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.mbarrier.arrive.nocomplete", .{
        .operands = .{ .flat = &.{ addr, count } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.mbarrier.arrive_drop`. Result types are explicit; attributes use mlir.Attribute.
pub fn mbarrier_arrive_drop(ctx: *mlir.Context, addr: *const mlir.Value, count: ?*const mlir.Value, multicastMask: ?*const mlir.Value, res_type: ?*const mlir.Type, scope: ?*const mlir.Attribute, relaxed: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var results: stdx.BoundedArray(*const mlir.Type, 256) = .empty;
    if (res_type) |value| results.appendAssumeCapacity(value);
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (scope) |value| attributes.appendAssumeCapacity(.named(ctx, "scope", value));
    if (relaxed) |value| attributes.appendAssumeCapacity(.named(ctx, "relaxed", value));
    return mlir.Operation.make(ctx, "nvvm.mbarrier.arrive_drop", .{
        .operands = .{ .variadic = &.{
            &.{addr},
            if (count) |value| &.{value} else &.{},
            if (multicastMask) |value| &.{value} else &.{},
        } },
        .results = .{ .flat = results.constSlice() },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mbarrier.arrive_drop.expect_tx`. Result types are explicit; attributes use mlir.Attribute.
pub fn mbarrier_arrive_drop_expect_tx(ctx: *mlir.Context, addr: *const mlir.Value, txcount: *const mlir.Value, multicastMask: ?*const mlir.Value, res_type: ?*const mlir.Type, scope: ?*const mlir.Attribute, relaxed: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var results: stdx.BoundedArray(*const mlir.Type, 256) = .empty;
    if (res_type) |value| results.appendAssumeCapacity(value);
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (scope) |value| attributes.appendAssumeCapacity(.named(ctx, "scope", value));
    if (relaxed) |value| attributes.appendAssumeCapacity(.named(ctx, "relaxed", value));
    return mlir.Operation.make(ctx, "nvvm.mbarrier.arrive_drop.expect_tx", .{
        .operands = .{ .flat = if (multicastMask) |value| &.{ addr, txcount, value } else &.{ addr, txcount } },
        .results = .{ .flat = results.constSlice() },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mbarrier.arrive_drop.nocomplete`. Result types are explicit; attributes use mlir.Attribute.
pub fn mbarrier_arrive_drop_nocomplete(ctx: *mlir.Context, addr: *const mlir.Value, count: *const mlir.Value, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.mbarrier.arrive_drop.nocomplete", .{
        .operands = .{ .flat = &.{ addr, count } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.mbarrier.check.layout`. Result types are explicit; attributes use mlir.Attribute.
pub fn mbarrier_check_layout(ctx: *mlir.Context, addr: *const mlir.Value, res_type: *const mlir.Type, layout: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.mbarrier.check.layout", .{
        .operands = .{ .flat = &.{addr} },
        .results = .{ .flat = &.{res_type} },
        .attributes = &.{
            .named(ctx, "layout", layout),
        },
        .location = location,
    });
}

/// `nvvm.mbarrier.complete_tx`. Result types are explicit; attributes use mlir.Attribute.
pub fn mbarrier_complete_tx(ctx: *mlir.Context, addr: *const mlir.Value, txcount: *const mlir.Value, report: ?*const mlir.Value, multicastMask: ?*const mlir.Value, scope: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (scope) |value| attributes.appendAssumeCapacity(.named(ctx, "scope", value));
    return mlir.Operation.make(ctx, "nvvm.mbarrier.complete_tx", .{
        .operands = .{ .variadic = &.{
            &.{addr},
            &.{txcount},
            if (report) |value| &.{value} else &.{},
            if (multicastMask) |value| &.{value} else &.{},
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mbarrier.expect_tx`. Result types are explicit; attributes use mlir.Attribute.
pub fn mbarrier_expect_tx(ctx: *mlir.Context, addr: *const mlir.Value, txcount: *const mlir.Value, multicastMask: ?*const mlir.Value, scope: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (scope) |value| attributes.appendAssumeCapacity(.named(ctx, "scope", value));
    return mlir.Operation.make(ctx, "nvvm.mbarrier.expect_tx", .{
        .operands = .{ .flat = if (multicastMask) |value| &.{ addr, txcount, value } else &.{ addr, txcount } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mbarrier.init`. Result types are explicit; attributes use mlir.Attribute.
pub fn mbarrier_init(ctx: *mlir.Context, addr: *const mlir.Value, count: *const mlir.Value, predicate: ?*const mlir.Value, layout: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (layout) |value| attributes.appendAssumeCapacity(.named(ctx, "layout", value));
    return mlir.Operation.make(ctx, "nvvm.mbarrier.init", .{
        .operands = .{ .flat = if (predicate) |value| &.{ addr, count, value } else &.{ addr, count } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mbarrier.inval`. Result types are explicit; attributes use mlir.Attribute.
pub fn mbarrier_inval(ctx: *mlir.Context, addr: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.mbarrier.inval", .{
        .operands = .{ .flat = &.{addr} },
        .location = location,
    });
}

/// `nvvm.mbarrier.test.wait`. Result types are explicit; attributes use mlir.Attribute.
pub fn mbarrier_test_wait(ctx: *mlir.Context, addr: *const mlir.Value, stateOrPhase: *const mlir.Value, res_type: *const mlir.Type, scope: ?*const mlir.Attribute, relaxed: ?*const mlir.Attribute, mbarrierPhase: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (scope) |value| attributes.appendAssumeCapacity(.named(ctx, "scope", value));
    if (relaxed) |value| attributes.appendAssumeCapacity(.named(ctx, "relaxed", value));
    if (mbarrierPhase) |value| attributes.appendAssumeCapacity(.named(ctx, "mbarrierPhase", value));
    return mlir.Operation.make(ctx, "nvvm.mbarrier.test.wait", .{
        .operands = .{ .flat = &.{ addr, stateOrPhase } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mbarrier.try_wait`. Result types are explicit; attributes use mlir.Attribute.
pub fn mbarrier_try_wait(ctx: *mlir.Context, addr: *const mlir.Value, stateOrPhase: *const mlir.Value, ticks: ?*const mlir.Value, res_type: *const mlir.Type, scope: ?*const mlir.Attribute, relaxed: ?*const mlir.Attribute, mbarrierPhase: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (scope) |value| attributes.appendAssumeCapacity(.named(ctx, "scope", value));
    if (relaxed) |value| attributes.appendAssumeCapacity(.named(ctx, "relaxed", value));
    if (mbarrierPhase) |value| attributes.appendAssumeCapacity(.named(ctx, "mbarrierPhase", value));
    return mlir.Operation.make(ctx, "nvvm.mbarrier.try_wait", .{
        .operands = .{ .flat = if (ticks) |value| &.{ addr, stateOrPhase, value } else &.{ addr, stateOrPhase } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mbarrier.try_wait.parity`. Result types are explicit; attributes use mlir.Attribute.
pub fn mbarrier_try_wait_parity(ctx: *mlir.Context, addr: *const mlir.Value, phase: *const mlir.Value, ticks: *const mlir.Value, useIntrinsic: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (useIntrinsic) |value| attributes.appendAssumeCapacity(.named(ctx, "useIntrinsic", value));
    return mlir.Operation.make(ctx, "nvvm.mbarrier.try_wait.parity", .{
        .operands = .{ .flat = &.{ addr, phase, ticks } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mbarrier.try_wait.parity.timelimit`. Result types are explicit; attributes use mlir.Attribute.
pub fn mbarrier_try_wait_parity_timelimit(ctx: *mlir.Context, addr: *const mlir.Value, phase: *const mlir.Value, timeLimit: *const mlir.Value, res_type: *const mlir.Type, scope: ?*const mlir.Attribute, order: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (scope) |value| attributes.appendAssumeCapacity(.named(ctx, "scope", value));
    if (order) |value| attributes.appendAssumeCapacity(.named(ctx, "order", value));
    return mlir.Operation.make(ctx, "nvvm.mbarrier.try_wait.parity.timelimit", .{
        .operands = .{ .flat = &.{ addr, phase, timeLimit } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mbarrier.try_wait.timelimit`. Result types are explicit; attributes use mlir.Attribute.
pub fn mbarrier_try_wait_timelimit(ctx: *mlir.Context, addr: *const mlir.Value, state: *const mlir.Value, timeLimit: *const mlir.Value, res_type: *const mlir.Type, scope: ?*const mlir.Attribute, order: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (scope) |value| attributes.appendAssumeCapacity(.named(ctx, "scope", value));
    if (order) |value| attributes.appendAssumeCapacity(.named(ctx, "order", value));
    return mlir.Operation.make(ctx, "nvvm.mbarrier.try_wait.timelimit", .{
        .operands = .{ .flat = &.{ addr, state, timeLimit } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mbarrier.txn`. Result types are explicit; attributes use mlir.Attribute.
pub fn mbarrier_txn(ctx: *mlir.Context, addr: *const mlir.Value, count: *const mlir.Value, multicast: ?*const mlir.Value, kind: *const mlir.Attribute, space: ?*const mlir.Attribute, scope: ?*const mlir.Attribute, order: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (space) |value| attributes.appendAssumeCapacity(.named(ctx, "space", value));
    if (scope) |value| attributes.appendAssumeCapacity(.named(ctx, "scope", value));
    if (order) |value| attributes.appendAssumeCapacity(.named(ctx, "order", value));
    return mlir.Operation.make(ctx, "nvvm.mbarrier.txn", .{
        .operands = .{ .flat = if (multicast) |value| &.{ addr, count, value } else &.{ addr, count } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mbarrier.txn.cta`. Result types are explicit; attributes use mlir.Attribute.
pub fn mbarrier_txn_cta(ctx: *mlir.Context, addr: *const mlir.Value, count: *const mlir.Value, res_type: *const mlir.Type, kind: *const mlir.Attribute, scope: ?*const mlir.Attribute, order: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (scope) |value| attributes.appendAssumeCapacity(.named(ctx, "scope", value));
    if (order) |value| attributes.appendAssumeCapacity(.named(ctx, "order", value));
    return mlir.Operation.make(ctx, "nvvm.mbarrier.txn.cta", .{
        .operands = .{ .flat = &.{ addr, count } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mbarrier.wait`. Result types are explicit; attributes use mlir.Attribute.
pub fn mbarrier_wait(ctx: *mlir.Context, addr: *const mlir.Value, state: *const mlir.Value, res_type: *const mlir.Type, kind: *const mlir.Attribute, scope: ?*const mlir.Attribute, order: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (scope) |value| attributes.appendAssumeCapacity(.named(ctx, "scope", value));
    if (order) |value| attributes.appendAssumeCapacity(.named(ctx, "order", value));
    return mlir.Operation.make(ctx, "nvvm.mbarrier.wait", .{
        .operands = .{ .flat = &.{ addr, state } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mbarrier.wait.parity`. Result types are explicit; attributes use mlir.Attribute.
pub fn mbarrier_wait_parity(ctx: *mlir.Context, addr: *const mlir.Value, phase: *const mlir.Value, res_type: *const mlir.Type, kind: *const mlir.Attribute, scope: ?*const mlir.Attribute, order: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (scope) |value| attributes.appendAssumeCapacity(.named(ctx, "scope", value));
    if (order) |value| attributes.appendAssumeCapacity(.named(ctx, "order", value));
    return mlir.Operation.make(ctx, "nvvm.mbarrier.wait.parity", .{
        .operands = .{ .flat = &.{ addr, phase } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.memory.barrier`. Result types are explicit; attributes use mlir.Attribute.
pub fn memory_barrier(ctx: *mlir.Context, scope: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.memory.barrier", .{
        .attributes = &.{
            .named(ctx, "scope", scope),
        },
        .location = location,
    });
}

/// `nvvm.mma.block_scale`. Result types are explicit; attributes use mlir.Attribute.
pub fn mma_block_scale(ctx: *mlir.Context, operandA: []const *const mlir.Value, operandB: []const *const mlir.Value, operandC: []const *const mlir.Value, scaleAData: *const mlir.Value, byteIdA: *const mlir.Value, threadIdA: *const mlir.Value, scaleBData: *const mlir.Value, byteIdB: *const mlir.Value, threadIdB: *const mlir.Value, res_type: *const mlir.Type, shape: *const mlir.Attribute, multiplicandAPtxType: ?*const mlir.Attribute, multiplicandBPtxType: ?*const mlir.Attribute, scaleVecSize: *const mlir.Attribute, blockScaleFormat: *const mlir.Attribute, kind: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 6) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "shape", shape));
    if (multiplicandAPtxType) |value| attributes.appendAssumeCapacity(.named(ctx, "multiplicandAPtxType", value));
    if (multiplicandBPtxType) |value| attributes.appendAssumeCapacity(.named(ctx, "multiplicandBPtxType", value));
    attributes.appendAssumeCapacity(.named(ctx, "scaleVecSize", scaleVecSize));
    attributes.appendAssumeCapacity(.named(ctx, "blockScaleFormat", blockScaleFormat));
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    return mlir.Operation.make(ctx, "nvvm.mma.block_scale", .{
        .operands = .{ .variadic = &.{
            operandA,
            operandB,
            operandC,
            &.{scaleAData},
            &.{byteIdA},
            &.{threadIdA},
            &.{scaleBData},
            &.{byteIdB},
            &.{threadIdB},
        } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mma.block_scale.internal`. Result types are explicit; attributes use mlir.Attribute.
pub fn mma_block_scale_internal(ctx: *mlir.Context, operandA: *const mlir.Value, scaleAData: *const mlir.Value, byteIdA: *const mlir.Value, threadIdA: *const mlir.Value, operandB: *const mlir.Value, scaleBData: *const mlir.Value, byteIdB: *const mlir.Value, threadIdB: *const mlir.Value, operandC: *const mlir.Value, res_type: *const mlir.Type, shape: *const mlir.Attribute, layoutA: *const mlir.Attribute, layoutB: *const mlir.Attribute, aType: *const mlir.Attribute, bType: *const mlir.Attribute, cType: *const mlir.Attribute, scaleVecSize: *const mlir.Attribute, blockScaleFormat: *const mlir.Attribute, kind: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 9) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "shape", shape));
    attributes.appendAssumeCapacity(.named(ctx, "layoutA", layoutA));
    attributes.appendAssumeCapacity(.named(ctx, "layoutB", layoutB));
    attributes.appendAssumeCapacity(.named(ctx, "aType", aType));
    attributes.appendAssumeCapacity(.named(ctx, "bType", bType));
    attributes.appendAssumeCapacity(.named(ctx, "cType", cType));
    attributes.appendAssumeCapacity(.named(ctx, "scaleVecSize", scaleVecSize));
    attributes.appendAssumeCapacity(.named(ctx, "blockScaleFormat", blockScaleFormat));
    if (kind) |value| attributes.appendAssumeCapacity(.named(ctx, "kind", value));
    return mlir.Operation.make(ctx, "nvvm.mma.block_scale.internal", .{
        .operands = .{ .flat = &.{ operandA, scaleAData, byteIdA, threadIdA, operandB, scaleBData, byteIdB, threadIdB, operandC } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mma.sp.block_scale`. Result types are explicit; attributes use mlir.Attribute.
pub fn mma_sp_block_scale(ctx: *mlir.Context, operandA: []const *const mlir.Value, operandB: []const *const mlir.Value, operandC: []const *const mlir.Value, sparseMetadata: *const mlir.Value, sparsitySelector: *const mlir.Value, scaleAData: *const mlir.Value, byteIdA: *const mlir.Value, threadIdA: *const mlir.Value, scaleBData: *const mlir.Value, byteIdB: *const mlir.Value, threadIdB: *const mlir.Value, res_type: *const mlir.Type, shape: *const mlir.Attribute, multiplicandAPtxType: ?*const mlir.Attribute, multiplicandBPtxType: ?*const mlir.Attribute, scaleVecSize: *const mlir.Attribute, blockScaleFormat: *const mlir.Attribute, kind: *const mlir.Attribute, orderedMetadata: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 7) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "shape", shape));
    if (multiplicandAPtxType) |value| attributes.appendAssumeCapacity(.named(ctx, "multiplicandAPtxType", value));
    if (multiplicandBPtxType) |value| attributes.appendAssumeCapacity(.named(ctx, "multiplicandBPtxType", value));
    attributes.appendAssumeCapacity(.named(ctx, "scaleVecSize", scaleVecSize));
    attributes.appendAssumeCapacity(.named(ctx, "blockScaleFormat", blockScaleFormat));
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (orderedMetadata) |value| attributes.appendAssumeCapacity(.named(ctx, "orderedMetadata", value));
    return mlir.Operation.make(ctx, "nvvm.mma.sp.block_scale", .{
        .operands = .{ .variadic = &.{
            operandA,
            operandB,
            operandC,
            &.{sparseMetadata},
            &.{sparsitySelector},
            &.{scaleAData},
            &.{byteIdA},
            &.{threadIdA},
            &.{scaleBData},
            &.{byteIdB},
            &.{threadIdB},
        } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mma.sp.block_scale.internal`. Result types are explicit; attributes use mlir.Attribute.
pub fn mma_sp_block_scale_internal(ctx: *mlir.Context, operandA: *const mlir.Value, scaleAData: *const mlir.Value, byteIdA: *const mlir.Value, threadIdA: *const mlir.Value, operandB: *const mlir.Value, scaleBData: *const mlir.Value, byteIdB: *const mlir.Value, threadIdB: *const mlir.Value, operandC: *const mlir.Value, sp_metadata: *const mlir.Value, selector: *const mlir.Value, res_type: *const mlir.Type, shape: *const mlir.Attribute, b1Op: ?*const mlir.Attribute, intOverflowBehavior: ?*const mlir.Attribute, layoutA: *const mlir.Attribute, layoutB: *const mlir.Attribute, aType: *const mlir.Attribute, bType: *const mlir.Attribute, cType: *const mlir.Attribute, scaleVecSize: *const mlir.Attribute, blockScaleFormat: *const mlir.Attribute, sparsityFormat: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 11) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "shape", shape));
    if (b1Op) |value| attributes.appendAssumeCapacity(.named(ctx, "b1Op", value));
    if (intOverflowBehavior) |value| attributes.appendAssumeCapacity(.named(ctx, "intOverflowBehavior", value));
    attributes.appendAssumeCapacity(.named(ctx, "layoutA", layoutA));
    attributes.appendAssumeCapacity(.named(ctx, "layoutB", layoutB));
    attributes.appendAssumeCapacity(.named(ctx, "aType", aType));
    attributes.appendAssumeCapacity(.named(ctx, "bType", bType));
    attributes.appendAssumeCapacity(.named(ctx, "cType", cType));
    attributes.appendAssumeCapacity(.named(ctx, "scaleVecSize", scaleVecSize));
    attributes.appendAssumeCapacity(.named(ctx, "blockScaleFormat", blockScaleFormat));
    attributes.appendAssumeCapacity(.named(ctx, "sparsityFormat", sparsityFormat));
    return mlir.Operation.make(ctx, "nvvm.mma.sp.block_scale.internal", .{
        .operands = .{ .flat = &.{ operandA, scaleAData, byteIdA, threadIdA, operandB, scaleBData, byteIdB, threadIdB, operandC, sp_metadata, selector } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mma.sp.sync`. Result types are explicit; attributes use mlir.Attribute.
pub fn mma_sp_sync(ctx: *mlir.Context, operandA: []const *const mlir.Value, operandB: []const *const mlir.Value, operandC: []const *const mlir.Value, sparseMetadata: *const mlir.Value, sparsitySelector: *const mlir.Value, res_type: *const mlir.Type, shape: *const mlir.Attribute, intOverflowBehavior: ?*const mlir.Attribute, multiplicandAPtxType: ?*const mlir.Attribute, multiplicandBPtxType: ?*const mlir.Attribute, orderedMetadata: ?*const mlir.Attribute, kind: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 6) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "shape", shape));
    if (intOverflowBehavior) |value| attributes.appendAssumeCapacity(.named(ctx, "intOverflowBehavior", value));
    if (multiplicandAPtxType) |value| attributes.appendAssumeCapacity(.named(ctx, "multiplicandAPtxType", value));
    if (multiplicandBPtxType) |value| attributes.appendAssumeCapacity(.named(ctx, "multiplicandBPtxType", value));
    if (orderedMetadata) |value| attributes.appendAssumeCapacity(.named(ctx, "orderedMetadata", value));
    if (kind) |value| attributes.appendAssumeCapacity(.named(ctx, "kind", value));
    return mlir.Operation.make(ctx, "nvvm.mma.sp.sync", .{
        .operands = .{ .variadic = &.{
            operandA,
            operandB,
            operandC,
            &.{sparseMetadata},
            &.{sparsitySelector},
        } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mma.sync`. Result types are explicit; attributes use mlir.Attribute.
pub fn mma_sync(ctx: *mlir.Context, operandA: []const *const mlir.Value, operandB: []const *const mlir.Value, operandC: []const *const mlir.Value, res_type: *const mlir.Type, shape: *const mlir.Attribute, b1Op: ?*const mlir.Attribute, intOverflowBehavior: ?*const mlir.Attribute, layoutA: *const mlir.Attribute, layoutB: *const mlir.Attribute, multiplicandAPtxType: ?*const mlir.Attribute, multiplicandBPtxType: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 7) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "shape", shape));
    if (b1Op) |value| attributes.appendAssumeCapacity(.named(ctx, "b1Op", value));
    if (intOverflowBehavior) |value| attributes.appendAssumeCapacity(.named(ctx, "intOverflowBehavior", value));
    attributes.appendAssumeCapacity(.named(ctx, "layoutA", layoutA));
    attributes.appendAssumeCapacity(.named(ctx, "layoutB", layoutB));
    if (multiplicandAPtxType) |value| attributes.appendAssumeCapacity(.named(ctx, "multiplicandAPtxType", value));
    if (multiplicandBPtxType) |value| attributes.appendAssumeCapacity(.named(ctx, "multiplicandBPtxType", value));
    return mlir.Operation.make(ctx, "nvvm.mma.sync", .{
        .operands = .{ .variadic = &.{
            operandA,
            operandB,
            operandC,
        } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mma_smem_desc`. Result types are explicit; attributes use mlir.Attribute.
pub fn mma_smem_desc(ctx: *mlir.Context, pointer: *const mlir.Value, ldm: *const mlir.Value, stride: *const mlir.Value, baseOffset: *const mlir.Value, swizzle: *const mlir.Value, res_type: *const mlir.Type, mmaDescVersion: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (mmaDescVersion) |value| attributes.appendAssumeCapacity(.named(ctx, "mmaDescVersion", value));
    return mlir.Operation.make(ctx, "nvvm.mma_smem_desc", .{
        .operands = .{ .flat = &.{ pointer, ldm, stride, baseOffset, swizzle } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.movmatrix`. Result types are explicit; attributes use mlir.Attribute.
pub fn movmatrix(ctx: *mlir.Context, src: *const mlir.Value, dst_type: *const mlir.Type, shape: *const mlir.Attribute, layout: ?*const mlir.Attribute, eltType: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "shape", shape));
    if (layout) |value| attributes.appendAssumeCapacity(.named(ctx, "layout", value));
    attributes.appendAssumeCapacity(.named(ctx, "eltType", eltType));
    return mlir.Operation.make(ctx, "nvvm.movmatrix", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mul`. Result types are explicit; attributes use mlir.Attribute.
pub fn mul(ctx: *mlir.Context, a: *const mlir.Value, b: *const mlir.Value, res_type: *const mlir.Type, mode: *const mlir.Attribute, isSigned: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "mode", mode));
    if (isSigned) |value| attributes.appendAssumeCapacity(.named(ctx, "isSigned", value));
    return mlir.Operation.make(ctx, "nvvm.mul", .{
        .operands = .{ .flat = &.{ a, b } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mul.packed.bf16x2.bf16x2.f16x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn mul_packed_bf16x2_bf16x2_f16x2(ctx: *mlir.Context, srcA: *const mlir.Value, srcB: *const mlir.Value, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.mul.packed.bf16x2.bf16x2.f16x2", .{
        .operands = .{ .flat = &.{ srcA, srcB } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.mul.packed.bf16x2.f32x2.f32x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn mul_packed_bf16x2_f32x2_f32x2(ctx: *mlir.Context, srcA: *const mlir.Value, srcB: *const mlir.Value, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.mul.packed.bf16x2.f32x2.f32x2", .{
        .operands = .{ .flat = &.{ srcA, srcB } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.mul.packed.f16x2.f16x2.bf16x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn mul_packed_f16x2_f16x2_bf16x2(ctx: *mlir.Context, srcA: *const mlir.Value, srcB: *const mlir.Value, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.mul.packed.f16x2.f16x2.bf16x2", .{
        .operands = .{ .flat = &.{ srcA, srcB } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.mul.packed.f16x2.f32x2.f32x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn mul_packed_f16x2_f32x2_f32x2(ctx: *mlir.Context, srcA: *const mlir.Value, srcB: *const mlir.Value, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.mul.packed.f16x2.f32x2.f32x2", .{
        .operands = .{ .flat = &.{ srcA, srcB } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.mul.packed.f32x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn mul_packed_f32x2(ctx: *mlir.Context, srcA: *const mlir.Value, srcB: *const mlir.Value, res_type: *const mlir.Type, rnd: ?*const mlir.Attribute, ftz: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (ftz) |value| attributes.appendAssumeCapacity(.named(ctx, "ftz", value));
    return mlir.Operation.make(ctx, "nvvm.mul.packed.f32x2", .{
        .operands = .{ .flat = &.{ srcA, srcB } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.multimem.ld.reduce`. Result types are explicit; attributes use mlir.Attribute.
pub fn multimem_ld_reduce(ctx: *mlir.Context, addr: *const mlir.Value, res_type: *const mlir.Type, ordering: *const mlir.Attribute, op: *const mlir.Attribute, dataType: *const mlir.Attribute, accPrec: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "ordering", ordering));
    attributes.appendAssumeCapacity(.named(ctx, "op", op));
    attributes.appendAssumeCapacity(.named(ctx, "dataType", dataType));
    if (accPrec) |value| attributes.appendAssumeCapacity(.named(ctx, "accPrec", value));
    return mlir.Operation.make(ctx, "nvvm.multimem.ld.reduce", .{
        .operands = .{ .flat = &.{addr} },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.multimem.red`. Result types are explicit; attributes use mlir.Attribute.
pub fn multimem_red(ctx: *mlir.Context, val: *const mlir.Value, addr: *const mlir.Value, ordering: *const mlir.Attribute, scope: *const mlir.Attribute, op: *const mlir.Attribute, dataType: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.multimem.red", .{
        .operands = .{ .flat = &.{ val, addr } },
        .attributes = &.{
            .named(ctx, "ordering", ordering),
            .named(ctx, "scope", scope),
            .named(ctx, "op", op),
            .named(ctx, "dataType", dataType),
        },
        .location = location,
    });
}

/// `nvvm.multimem.st`. Result types are explicit; attributes use mlir.Attribute.
pub fn multimem_st(ctx: *mlir.Context, val: *const mlir.Value, addr: *const mlir.Value, ordering: *const mlir.Attribute, scope: *const mlir.Attribute, dataType: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.multimem.st", .{
        .operands = .{ .flat = &.{ val, addr } },
        .attributes = &.{
            .named(ctx, "ordering", ordering),
            .named(ctx, "scope", scope),
            .named(ctx, "dataType", dataType),
        },
        .location = location,
    });
}

/// `nvvm.nanosleep`. Result types are explicit; attributes use mlir.Attribute.
pub fn nanosleep(ctx: *mlir.Context, duration: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.nanosleep", .{
        .operands = .{ .flat = &.{duration} },
        .location = location,
    });
}

/// `nvvm.pmevent`. Result types are explicit; attributes use mlir.Attribute.
pub fn pmevent(ctx: *mlir.Context, maskedEventId: ?*const mlir.Attribute, eventId: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (maskedEventId) |value| attributes.appendAssumeCapacity(.named(ctx, "maskedEventId", value));
    if (eventId) |value| attributes.appendAssumeCapacity(.named(ctx, "eventId", value));
    return mlir.Operation.make(ctx, "nvvm.pmevent", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.prefetch`. Result types are explicit; attributes use mlir.Attribute.
pub fn prefetch(ctx: *mlir.Context, addr: *const mlir.Value, predicate: ?*const mlir.Value, cacheLevel: ?*const mlir.Attribute, evictPriority: ?*const mlir.Attribute, tensormap: ?*const mlir.Attribute, uniform: ?*const mlir.Attribute, in_param_space: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    if (cacheLevel) |value| attributes.appendAssumeCapacity(.named(ctx, "cacheLevel", value));
    if (evictPriority) |value| attributes.appendAssumeCapacity(.named(ctx, "evictPriority", value));
    if (tensormap) |value| attributes.appendAssumeCapacity(.named(ctx, "tensormap", value));
    if (uniform) |value| attributes.appendAssumeCapacity(.named(ctx, "uniform", value));
    if (in_param_space) |value| attributes.appendAssumeCapacity(.named(ctx, "in_param_space", value));
    return mlir.Operation.make(ctx, "nvvm.prefetch", .{
        .operands = .{ .flat = if (predicate) |value| &.{ addr, value } else &.{addr} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.prmt`. Result types are explicit; attributes use mlir.Attribute.
pub fn prmt(ctx: *mlir.Context, lo: *const mlir.Value, hi: ?*const mlir.Value, selector: *const mlir.Value, res_type: *const mlir.Type, mode: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.prmt", .{
        .operands = .{ .flat = if (hi) |value| &.{ lo, value, selector } else &.{ lo, selector } },
        .results = .{ .flat = &.{res_type} },
        .attributes = &.{
            .named(ctx, "mode", mode),
        },
        .location = location,
    });
}

/// `nvvm.rcp.approx.ftz.f`. Result types are explicit; attributes use mlir.Attribute.
pub fn rcp_approx_ftz_f(ctx: *mlir.Context, arg: *const mlir.Value, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.rcp.approx.ftz.f", .{
        .operands = .{ .flat = &.{arg} },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.aggr.smem.size`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_aggr_smem_size(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.aggr.smem.size", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.clock`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_clock(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.clock", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.clock64`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_clock64(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.clock64", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.cluster.ctaid.x`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_cluster_ctaid_x(ctx: *mlir.Context, res_type: *const mlir.Type, range: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.cluster.ctaid.x", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.cluster.ctaid.y`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_cluster_ctaid_y(ctx: *mlir.Context, res_type: *const mlir.Type, range: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.cluster.ctaid.y", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.cluster.ctaid.z`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_cluster_ctaid_z(ctx: *mlir.Context, res_type: *const mlir.Type, range: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.cluster.ctaid.z", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.cluster.ctarank`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_cluster_ctarank(ctx: *mlir.Context, res_type: *const mlir.Type, range: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.cluster.ctarank", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.cluster.nctaid.x`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_cluster_nctaid_x(ctx: *mlir.Context, res_type: *const mlir.Type, range: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.cluster.nctaid.x", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.cluster.nctaid.y`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_cluster_nctaid_y(ctx: *mlir.Context, res_type: *const mlir.Type, range: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.cluster.nctaid.y", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.cluster.nctaid.z`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_cluster_nctaid_z(ctx: *mlir.Context, res_type: *const mlir.Type, range: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.cluster.nctaid.z", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.cluster.nctarank`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_cluster_nctarank(ctx: *mlir.Context, res_type: *const mlir.Type, range: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.cluster.nctarank", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.clusterid.x`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_clusterid_x(ctx: *mlir.Context, res_type: *const mlir.Type, range: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.clusterid.x", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.clusterid.y`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_clusterid_y(ctx: *mlir.Context, res_type: *const mlir.Type, range: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.clusterid.y", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.clusterid.z`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_clusterid_z(ctx: *mlir.Context, res_type: *const mlir.Type, range: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.clusterid.z", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.ctaid.x`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_ctaid_x(ctx: *mlir.Context, res_type: *const mlir.Type, range: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.ctaid.x", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.ctaid.y`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_ctaid_y(ctx: *mlir.Context, res_type: *const mlir.Type, range: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.ctaid.y", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.ctaid.z`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_ctaid_z(ctx: *mlir.Context, res_type: *const mlir.Type, range: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.ctaid.z", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.dynamic.smem.size`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_dynamic_smem_size(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.dynamic.smem.size", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.envreg0`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_envreg0(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.envreg0", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.envreg1`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_envreg1(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.envreg1", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.envreg10`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_envreg10(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.envreg10", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.envreg11`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_envreg11(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.envreg11", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.envreg12`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_envreg12(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.envreg12", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.envreg13`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_envreg13(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.envreg13", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.envreg14`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_envreg14(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.envreg14", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.envreg15`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_envreg15(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.envreg15", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.envreg16`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_envreg16(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.envreg16", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.envreg17`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_envreg17(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.envreg17", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.envreg18`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_envreg18(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.envreg18", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.envreg19`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_envreg19(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.envreg19", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.envreg2`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_envreg2(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.envreg2", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.envreg20`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_envreg20(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.envreg20", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.envreg21`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_envreg21(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.envreg21", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.envreg22`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_envreg22(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.envreg22", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.envreg23`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_envreg23(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.envreg23", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.envreg24`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_envreg24(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.envreg24", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.envreg25`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_envreg25(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.envreg25", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.envreg26`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_envreg26(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.envreg26", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.envreg27`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_envreg27(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.envreg27", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.envreg28`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_envreg28(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.envreg28", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.envreg29`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_envreg29(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.envreg29", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.envreg3`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_envreg3(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.envreg3", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.envreg30`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_envreg30(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.envreg30", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.envreg31`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_envreg31(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.envreg31", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.envreg4`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_envreg4(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.envreg4", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.envreg5`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_envreg5(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.envreg5", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.envreg6`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_envreg6(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.envreg6", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.envreg7`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_envreg7(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.envreg7", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.envreg8`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_envreg8(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.envreg8", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.envreg9`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_envreg9(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.envreg9", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.globaltimer`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_globaltimer(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.globaltimer", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.globaltimer.lo`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_globaltimer_lo(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.globaltimer.lo", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.gridid`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_gridid(ctx: *mlir.Context, res_type: *const mlir.Type, range: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.gridid", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.laneid`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_laneid(ctx: *mlir.Context, res_type: *const mlir.Type, range: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.laneid", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.lanemask.eq`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_lanemask_eq(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.lanemask.eq", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.lanemask.ge`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_lanemask_ge(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.lanemask.ge", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.lanemask.gt`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_lanemask_gt(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.lanemask.gt", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.lanemask.le`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_lanemask_le(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.lanemask.le", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.lanemask.lt`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_lanemask_lt(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.lanemask.lt", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.nclusterid.x`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_nclusterid_x(ctx: *mlir.Context, res_type: *const mlir.Type, range: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.nclusterid.x", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.nclusterid.y`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_nclusterid_y(ctx: *mlir.Context, res_type: *const mlir.Type, range: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.nclusterid.y", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.nclusterid.z`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_nclusterid_z(ctx: *mlir.Context, res_type: *const mlir.Type, range: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.nclusterid.z", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.nctaid.x`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_nctaid_x(ctx: *mlir.Context, res_type: *const mlir.Type, range: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.nctaid.x", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.nctaid.y`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_nctaid_y(ctx: *mlir.Context, res_type: *const mlir.Type, range: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.nctaid.y", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.nctaid.z`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_nctaid_z(ctx: *mlir.Context, res_type: *const mlir.Type, range: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.nctaid.z", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.nsmid`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_nsmid(ctx: *mlir.Context, res_type: *const mlir.Type, range: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.nsmid", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.ntid.x`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_ntid_x(ctx: *mlir.Context, res_type: *const mlir.Type, range: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.ntid.x", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.ntid.y`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_ntid_y(ctx: *mlir.Context, res_type: *const mlir.Type, range: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.ntid.y", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.ntid.z`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_ntid_z(ctx: *mlir.Context, res_type: *const mlir.Type, range: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.ntid.z", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.nwarpid`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_nwarpid(ctx: *mlir.Context, res_type: *const mlir.Type, range: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.nwarpid", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.smid`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_smid(ctx: *mlir.Context, res_type: *const mlir.Type, range: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.smid", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.tid.x`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_tid_x(ctx: *mlir.Context, res_type: *const mlir.Type, range: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.tid.x", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.tid.y`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_tid_y(ctx: *mlir.Context, res_type: *const mlir.Type, range: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.tid.y", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.tid.z`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_tid_z(ctx: *mlir.Context, res_type: *const mlir.Type, range: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.tid.z", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.total.smem.size`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_total_smem_size(ctx: *mlir.Context, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.total.smem.size", .{
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.warpid`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_warpid(ctx: *mlir.Context, res_type: *const mlir.Type, range: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.warpid", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.warpsize`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_warpsize(ctx: *mlir.Context, res_type: *const mlir.Type, range: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.warpsize", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.red`. Result types are explicit; attributes use mlir.Attribute.
pub fn red(ctx: *mlir.Context, a: *const mlir.Value, b: *const mlir.Value, cacheHint: ?*const mlir.Value, memOrder: ?*const mlir.Attribute, memScope: ?*const mlir.Attribute, op: *const mlir.Attribute, type_: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    if (memOrder) |value| attributes.appendAssumeCapacity(.named(ctx, "memOrder", value));
    if (memScope) |value| attributes.appendAssumeCapacity(.named(ctx, "memScope", value));
    attributes.appendAssumeCapacity(.named(ctx, "op", op));
    attributes.appendAssumeCapacity(.named(ctx, "type", type_));
    return mlir.Operation.make(ctx, "nvvm.red", .{
        .operands = .{ .flat = if (cacheHint) |value| &.{ a, b, value } else &.{ a, b } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.redux.sync`. Result types are explicit; attributes use mlir.Attribute.
pub fn redux_sync(ctx: *mlir.Context, val: *const mlir.Value, mask_and_clamp: *const mlir.Value, res_type: *const mlir.Type, kind: *const mlir.Attribute, abs: ?*const mlir.Attribute, nan: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (abs) |value| attributes.appendAssumeCapacity(.named(ctx, "abs", value));
    if (nan) |value| attributes.appendAssumeCapacity(.named(ctx, "nan", value));
    return mlir.Operation.make(ctx, "nvvm.redux.sync", .{
        .operands = .{ .flat = &.{ val, mask_and_clamp } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.rsqrt`. Result types are explicit; attributes use mlir.Attribute.
pub fn rsqrt(ctx: *mlir.Context, src: *const mlir.Value, res_type: *const mlir.Type, ftz: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (ftz) |value| attributes.appendAssumeCapacity(.named(ctx, "ftz", value));
    return mlir.Operation.make(ctx, "nvvm.rsqrt", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.setmaxregister`. Result types are explicit; attributes use mlir.Attribute.
pub fn setmaxregister(ctx: *mlir.Context, regCount: *const mlir.Attribute, action: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.setmaxregister", .{
        .attributes = &.{
            .named(ctx, "regCount", regCount),
            .named(ctx, "action", action),
        },
        .location = location,
    });
}

/// `nvvm.shfl.sync`. Result types are explicit; attributes use mlir.Attribute.
pub fn shfl_sync(ctx: *mlir.Context, thread_mask: *const mlir.Value, val: *const mlir.Value, offset: *const mlir.Value, mask_and_clamp: *const mlir.Value, res_type: *const mlir.Type, kind: *const mlir.Attribute, return_value_and_is_valid: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (return_value_and_is_valid) |value| attributes.appendAssumeCapacity(.named(ctx, "return_value_and_is_valid", value));
    return mlir.Operation.make(ctx, "nvvm.shfl.sync", .{
        .operands = .{ .flat = &.{ thread_mask, val, offset, mask_and_clamp } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.sin`. Result types are explicit; attributes use mlir.Attribute.
pub fn sin(ctx: *mlir.Context, src: *const mlir.Value, res_type: *const mlir.Type, ftz: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (ftz) |value| attributes.appendAssumeCapacity(.named(ctx, "ftz", value));
    return mlir.Operation.make(ctx, "nvvm.sin", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.spcompress`. Result types are explicit; attributes use mlir.Attribute.
pub fn spcompress(ctx: *mlir.Context, spDesc: *const mlir.Value, data: *const mlir.Value, metadata_type: *const mlir.Type, compressed_data_type: *const mlir.Type, compFactor: *const mlir.Attribute, indexSize: *const mlir.Attribute, elemSize: *const mlir.Attribute, repFactor: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.spcompress", .{
        .operands = .{ .flat = &.{ spDesc, data } },
        .results = .{ .flat = &.{ metadata_type, compressed_data_type } },
        .attributes = &.{
            .named(ctx, "compFactor", compFactor),
            .named(ctx, "indexSize", indexSize),
            .named(ctx, "elemSize", elemSize),
            .named(ctx, "repFactor", repFactor),
        },
        .location = location,
    });
}

/// `nvvm.spdecompress`. Result types are explicit; attributes use mlir.Attribute.
pub fn spdecompress(ctx: *mlir.Context, metadata: *const mlir.Value, compressed_data: *const mlir.Value, data_type: *const mlir.Type, compFactor: *const mlir.Attribute, indexSize: *const mlir.Attribute, elemSize: *const mlir.Attribute, repFactor: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.spdecompress", .{
        .operands = .{ .flat = &.{ metadata, compressed_data } },
        .results = .{ .flat = &.{data_type} },
        .attributes = &.{
            .named(ctx, "compFactor", compFactor),
            .named(ctx, "indexSize", indexSize),
            .named(ctx, "elemSize", elemSize),
            .named(ctx, "repFactor", repFactor),
        },
        .location = location,
    });
}

/// `nvvm.st.bulk`. Result types are explicit; attributes use mlir.Attribute.
pub fn st_bulk(ctx: *mlir.Context, addr: *const mlir.Value, size: *const mlir.Value, initVal: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (initVal) |value| attributes.appendAssumeCapacity(.named(ctx, "initVal", value));
    return mlir.Operation.make(ctx, "nvvm.st.bulk", .{
        .operands = .{ .flat = &.{ addr, size } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.stmatrix`. Result types are explicit; attributes use mlir.Attribute.
pub fn stmatrix(ctx: *mlir.Context, ptr: *const mlir.Value, sources: []const *const mlir.Value, layout: *const mlir.Attribute, shape: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 256) = .empty;
    operands.appendAssumeCapacity(ptr);
    operands.appendSliceAssumeCapacity(sources);
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "layout", layout));
    if (shape) |value| attributes.appendAssumeCapacity(.named(ctx, "shape", value));
    return mlir.Operation.make(ctx, "nvvm.stmatrix", .{
        .operands = .{ .flat = operands.constSlice() },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.store.async`. Result types are explicit; attributes use mlir.Attribute.
pub fn store_async(ctx: *mlir.Context, addr: *const mlir.Value, value_: *const mlir.Value, mbarrier: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.store.async", .{
        .operands = .{ .flat = &.{ addr, value_, mbarrier } },
        .location = location,
    });
}

/// `nvvm.store.ext`. Result types are explicit; attributes use mlir.Attribute.
pub fn store_ext(ctx: *mlir.Context, value_: *const mlir.Value, addr: *const mlir.Value, l2CacheHint: ?*const mlir.Value, order: ?*const mlir.Attribute, scope: ?*const mlir.Attribute, evict: ?*const mlir.Attribute, cacheModifier: ?*const mlir.Attribute, sharedSpace: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    if (order) |value| attributes.appendAssumeCapacity(.named(ctx, "order", value));
    if (scope) |value| attributes.appendAssumeCapacity(.named(ctx, "scope", value));
    if (evict) |value| attributes.appendAssumeCapacity(.named(ctx, "evict", value));
    if (cacheModifier) |value| attributes.appendAssumeCapacity(.named(ctx, "cacheModifier", value));
    if (sharedSpace) |value| attributes.appendAssumeCapacity(.named(ctx, "sharedSpace", value));
    return mlir.Operation.make(ctx, "nvvm.store.ext", .{
        .operands = .{ .flat = if (l2CacheHint) |value| &.{ value_, addr, value } else &.{ value_, addr } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.sub.packed.bf16x2.f32x2.f32x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn sub_packed_bf16x2_f32x2_f32x2(ctx: *mlir.Context, srcA: *const mlir.Value, srcB: *const mlir.Value, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.sub.packed.bf16x2.f32x2.f32x2", .{
        .operands = .{ .flat = &.{ srcA, srcB } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.sub.packed.f16x2.f32x2.f32x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn sub_packed_f16x2_f32x2_f32x2(ctx: *mlir.Context, srcA: *const mlir.Value, srcB: *const mlir.Value, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.sub.packed.f16x2.f32x2.f32x2", .{
        .operands = .{ .flat = &.{ srcA, srcB } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.sub.packed.f32x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn sub_packed_f32x2(ctx: *mlir.Context, srcA: *const mlir.Value, srcB: *const mlir.Value, res_type: *const mlir.Type, rnd: ?*const mlir.Attribute, ftz: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (ftz) |value| attributes.appendAssumeCapacity(.named(ctx, "ftz", value));
    return mlir.Operation.make(ctx, "nvvm.sub.packed.f32x2", .{
        .operands = .{ .flat = &.{ srcA, srcB } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.sub.packed.f32x2.bf16x2.f32x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn sub_packed_f32x2_bf16x2_f32x2(ctx: *mlir.Context, srcA: *const mlir.Value, srcB: *const mlir.Value, res_type: *const mlir.Type, rnd: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    return mlir.Operation.make(ctx, "nvvm.sub.packed.f32x2.bf16x2.f32x2", .{
        .operands = .{ .flat = &.{ srcA, srcB } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.sub.packed.f32x2.f16x2.f32x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn sub_packed_f32x2_f16x2_f32x2(ctx: *mlir.Context, srcA: *const mlir.Value, srcB: *const mlir.Value, res_type: *const mlir.Type, rnd: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    return mlir.Operation.make(ctx, "nvvm.sub.packed.f32x2.f16x2.f32x2", .{
        .operands = .{ .flat = &.{ srcA, srcB } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.subf`. Result types are explicit; attributes use mlir.Attribute.
pub fn subf(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, res_type: *const mlir.Type, rnd: ?*const mlir.Attribute, sat: ?*const mlir.Attribute, ftz: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (sat) |value| attributes.appendAssumeCapacity(.named(ctx, "sat", value));
    if (ftz) |value| attributes.appendAssumeCapacity(.named(ctx, "ftz", value));
    return mlir.Operation.make(ctx, "nvvm.subf", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.alloc`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_alloc(ctx: *mlir.Context, addr: *const mlir.Value, nCols: *const mlir.Value, isExclusive: ?*const mlir.Attribute, group: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (isExclusive) |value| attributes.appendAssumeCapacity(.named(ctx, "isExclusive", value));
    if (group) |value| attributes.appendAssumeCapacity(.named(ctx, "group", value));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.alloc", .{
        .operands = .{ .flat = &.{ addr, nCols } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.commit`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_commit(ctx: *mlir.Context, addr: *const mlir.Value, multicastMask: ?*const mlir.Value, smem_a_read: ?*const mlir.Attribute, group: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (smem_a_read) |value| attributes.appendAssumeCapacity(.named(ctx, "smem_a_read", value));
    if (group) |value| attributes.appendAssumeCapacity(.named(ctx, "group", value));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.commit", .{
        .operands = .{ .flat = if (multicastMask) |value| &.{ addr, value } else &.{addr} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.cp`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_cp(ctx: *mlir.Context, taddr: *const mlir.Value, smem_desc: *const mlir.Value, shape: *const mlir.Attribute, group: ?*const mlir.Attribute, multicast: ?*const mlir.Attribute, srcFormat: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "shape", shape));
    if (group) |value| attributes.appendAssumeCapacity(.named(ctx, "group", value));
    if (multicast) |value| attributes.appendAssumeCapacity(.named(ctx, "multicast", value));
    if (srcFormat) |value| attributes.appendAssumeCapacity(.named(ctx, "srcFormat", value));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.cp", .{
        .operands = .{ .flat = &.{ taddr, smem_desc } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.dealloc`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_dealloc(ctx: *mlir.Context, taddr: *const mlir.Value, nCols: *const mlir.Value, isExclusive: ?*const mlir.Attribute, group: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (isExclusive) |value| attributes.appendAssumeCapacity(.named(ctx, "isExclusive", value));
    if (group) |value| attributes.appendAssumeCapacity(.named(ctx, "group", value));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.dealloc", .{
        .operands = .{ .flat = &.{ taddr, nCols } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.fence`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_fence(ctx: *mlir.Context, kind: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.tcgen05.fence", .{
        .attributes = &.{
            .named(ctx, "kind", kind),
        },
        .location = location,
    });
}

/// `nvvm.tcgen05.ld`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_ld(ctx: *mlir.Context, tmemAddr: *const mlir.Value, offset: ?*const mlir.Value, res_type: *const mlir.Type, pack: ?*const mlir.Attribute, shape: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (pack) |value| attributes.appendAssumeCapacity(.named(ctx, "pack", value));
    attributes.appendAssumeCapacity(.named(ctx, "shape", shape));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.ld", .{
        .operands = .{ .flat = if (offset) |value| &.{ tmemAddr, value } else &.{tmemAddr} },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.ld.red`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_ld_red(ctx: *mlir.Context, addr: *const mlir.Value, offset: ?*const mlir.Value, data_type: *const mlir.Type, redVal_type: *const mlir.Type, shape: *const mlir.Attribute, op: *const mlir.Attribute, abs: ?*const mlir.Attribute, nan: ?*const mlir.Attribute, isSigned: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "shape", shape));
    attributes.appendAssumeCapacity(.named(ctx, "op", op));
    if (abs) |value| attributes.appendAssumeCapacity(.named(ctx, "abs", value));
    if (nan) |value| attributes.appendAssumeCapacity(.named(ctx, "nan", value));
    if (isSigned) |value| attributes.appendAssumeCapacity(.named(ctx, "isSigned", value));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.ld.red", .{
        .operands = .{ .flat = if (offset) |value| &.{ addr, value } else &.{addr} },
        .results = .{ .flat = &.{ data_type, redVal_type } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.ld.red.spcompress`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_ld_red_spcompress(ctx: *mlir.Context, tmemAddr: *const mlir.Value, metadata_type: *const mlir.Type, compressedData_type: *const mlir.Type, redVal_type: *const mlir.Type, abs: ?*const mlir.Attribute, nan: ?*const mlir.Attribute, shape: *const mlir.Attribute, opKind: *const mlir.Attribute, compFactor: *const mlir.Attribute, repFactor: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 6) = .empty;
    if (abs) |value| attributes.appendAssumeCapacity(.named(ctx, "abs", value));
    if (nan) |value| attributes.appendAssumeCapacity(.named(ctx, "nan", value));
    attributes.appendAssumeCapacity(.named(ctx, "shape", shape));
    attributes.appendAssumeCapacity(.named(ctx, "opKind", opKind));
    attributes.appendAssumeCapacity(.named(ctx, "compFactor", compFactor));
    attributes.appendAssumeCapacity(.named(ctx, "repFactor", repFactor));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.ld.red.spcompress", .{
        .operands = .{ .flat = &.{tmemAddr} },
        .results = .{ .flat = &.{ metadata_type, compressedData_type, redVal_type } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.ld.spcompress`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_ld_spcompress(ctx: *mlir.Context, tmemAddr: *const mlir.Value, metadata_type: *const mlir.Type, compressedData_type: *const mlir.Type, abs: ?*const mlir.Attribute, shape: *const mlir.Attribute, opKind: *const mlir.Attribute, compFactor: *const mlir.Attribute, repFactor: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    if (abs) |value| attributes.appendAssumeCapacity(.named(ctx, "abs", value));
    attributes.appendAssumeCapacity(.named(ctx, "shape", shape));
    attributes.appendAssumeCapacity(.named(ctx, "opKind", opKind));
    attributes.appendAssumeCapacity(.named(ctx, "compFactor", compFactor));
    attributes.appendAssumeCapacity(.named(ctx, "repFactor", repFactor));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.ld.spcompress", .{
        .operands = .{ .flat = &.{tmemAddr} },
        .results = .{ .flat = &.{ metadata_type, compressedData_type } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.mma`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_mma(ctx: *mlir.Context, matrixD: *const mlir.Value, matrixA: *const mlir.Value, matrixB: *const mlir.Value, idesc: *const mlir.Value, enableInputD: *const mlir.Value, scaleInputD: ?*const mlir.Value, disableOutputLane: ?*const mlir.Value, kind: *const mlir.Attribute, ctaGroup: *const mlir.Attribute, collectorOp: ?*const mlir.Attribute, collectorOpB: ?*const mlir.Attribute, aShift: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    attributes.appendAssumeCapacity(.named(ctx, "ctaGroup", ctaGroup));
    if (collectorOp) |value| attributes.appendAssumeCapacity(.named(ctx, "collectorOp", value));
    if (collectorOpB) |value| attributes.appendAssumeCapacity(.named(ctx, "collectorOpB", value));
    if (aShift) |value| attributes.appendAssumeCapacity(.named(ctx, "aShift", value));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.mma", .{
        .operands = .{ .variadic = &.{
            &.{matrixD},
            &.{matrixA},
            &.{matrixB},
            &.{idesc},
            &.{enableInputD},
            if (scaleInputD) |value| &.{value} else &.{},
            if (disableOutputLane) |value| &.{value} else &.{},
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.mma.block_scale`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_mma_block_scale(ctx: *mlir.Context, matrixD: *const mlir.Value, matrixA: *const mlir.Value, matrixB: *const mlir.Value, idesc: *const mlir.Value, enableInputD: *const mlir.Value, scaleA: *const mlir.Value, scaleB: *const mlir.Value, kind: *const mlir.Attribute, ctaGroup: *const mlir.Attribute, blockScale: ?*const mlir.Attribute, collectorOp: ?*const mlir.Attribute, collectorOpB: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    attributes.appendAssumeCapacity(.named(ctx, "ctaGroup", ctaGroup));
    if (blockScale) |value| attributes.appendAssumeCapacity(.named(ctx, "blockScale", value));
    if (collectorOp) |value| attributes.appendAssumeCapacity(.named(ctx, "collectorOp", value));
    if (collectorOpB) |value| attributes.appendAssumeCapacity(.named(ctx, "collectorOpB", value));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.mma.block_scale", .{
        .operands = .{ .flat = &.{ matrixD, matrixA, matrixB, idesc, enableInputD, scaleA, scaleB } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.mma.block_scale.decompress_b`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_mma_block_scale_decompress_b(ctx: *mlir.Context, matrixD: *const mlir.Value, matrixA: *const mlir.Value, matrixB: *const mlir.Value, idesc: *const mlir.Value, enableInputD: *const mlir.Value, scaleA: *const mlir.Value, scaleB: *const mlir.Value, decompressBMetadata: *const mlir.Value, kind: *const mlir.Attribute, ctaGroup: *const mlir.Attribute, blockScale: ?*const mlir.Attribute, collectorOpA: ?*const mlir.Attribute, collectorOpB: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    attributes.appendAssumeCapacity(.named(ctx, "ctaGroup", ctaGroup));
    if (blockScale) |value| attributes.appendAssumeCapacity(.named(ctx, "blockScale", value));
    if (collectorOpA) |value| attributes.appendAssumeCapacity(.named(ctx, "collectorOpA", value));
    if (collectorOpB) |value| attributes.appendAssumeCapacity(.named(ctx, "collectorOpB", value));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.mma.block_scale.decompress_b", .{
        .operands = .{ .flat = &.{ matrixD, matrixA, matrixB, idesc, enableInputD, scaleA, scaleB, decompressBMetadata } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.mma.decompress_b`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_mma_decompress_b(ctx: *mlir.Context, matrixD: *const mlir.Value, matrixA: *const mlir.Value, matrixB: *const mlir.Value, idesc: *const mlir.Value, enableInputD: *const mlir.Value, decompressBMetadata: *const mlir.Value, disableOutputLane: ?*const mlir.Value, kind: *const mlir.Attribute, ctaGroup: *const mlir.Attribute, collectorOpA: ?*const mlir.Attribute, collectorOpB: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    attributes.appendAssumeCapacity(.named(ctx, "ctaGroup", ctaGroup));
    if (collectorOpA) |value| attributes.appendAssumeCapacity(.named(ctx, "collectorOpA", value));
    if (collectorOpB) |value| attributes.appendAssumeCapacity(.named(ctx, "collectorOpB", value));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.mma.decompress_b", .{
        .operands = .{ .flat = if (disableOutputLane) |value| &.{ matrixD, matrixA, matrixB, idesc, enableInputD, decompressBMetadata, value } else &.{ matrixD, matrixA, matrixB, idesc, enableInputD, decompressBMetadata } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.mma.sp`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_mma_sp(ctx: *mlir.Context, matrixD: *const mlir.Value, matrixA: *const mlir.Value, matrixB: *const mlir.Value, idesc: *const mlir.Value, enableInputD: *const mlir.Value, sparseMetadata: *const mlir.Value, scaleInputD: ?*const mlir.Value, disableOutputLane: ?*const mlir.Value, kind: *const mlir.Attribute, ctaGroup: *const mlir.Attribute, collectorOp: ?*const mlir.Attribute, collectorOpB: ?*const mlir.Attribute, aShift: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    attributes.appendAssumeCapacity(.named(ctx, "ctaGroup", ctaGroup));
    if (collectorOp) |value| attributes.appendAssumeCapacity(.named(ctx, "collectorOp", value));
    if (collectorOpB) |value| attributes.appendAssumeCapacity(.named(ctx, "collectorOpB", value));
    if (aShift) |value| attributes.appendAssumeCapacity(.named(ctx, "aShift", value));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.mma.sp", .{
        .operands = .{ .variadic = &.{
            &.{matrixD},
            &.{matrixA},
            &.{matrixB},
            &.{idesc},
            &.{enableInputD},
            &.{sparseMetadata},
            if (scaleInputD) |value| &.{value} else &.{},
            if (disableOutputLane) |value| &.{value} else &.{},
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.mma.sp.block_scale`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_mma_sp_block_scale(ctx: *mlir.Context, matrixD: *const mlir.Value, matrixA: *const mlir.Value, matrixB: *const mlir.Value, idesc: *const mlir.Value, enableInputD: *const mlir.Value, sparseMetadata: *const mlir.Value, scaleA: *const mlir.Value, scaleB: *const mlir.Value, kind: *const mlir.Attribute, ctaGroup: *const mlir.Attribute, blockScale: ?*const mlir.Attribute, collectorOp: ?*const mlir.Attribute, collectorOpB: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    attributes.appendAssumeCapacity(.named(ctx, "ctaGroup", ctaGroup));
    if (blockScale) |value| attributes.appendAssumeCapacity(.named(ctx, "blockScale", value));
    if (collectorOp) |value| attributes.appendAssumeCapacity(.named(ctx, "collectorOp", value));
    if (collectorOpB) |value| attributes.appendAssumeCapacity(.named(ctx, "collectorOpB", value));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.mma.sp.block_scale", .{
        .operands = .{ .flat = &.{ matrixD, matrixA, matrixB, idesc, enableInputD, sparseMetadata, scaleA, scaleB } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.mma.ws`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_mma_ws(ctx: *mlir.Context, matrixD: *const mlir.Value, matrixA: *const mlir.Value, matrixB: *const mlir.Value, idesc: *const mlir.Value, enableInputD: *const mlir.Value, zeroColMask: ?*const mlir.Value, kind: *const mlir.Attribute, collectorBBuffer: ?*const mlir.Attribute, collectorOp: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (collectorBBuffer) |value| attributes.appendAssumeCapacity(.named(ctx, "collectorBBuffer", value));
    if (collectorOp) |value| attributes.appendAssumeCapacity(.named(ctx, "collectorOp", value));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.mma.ws", .{
        .operands = .{ .flat = if (zeroColMask) |value| &.{ matrixD, matrixA, matrixB, idesc, enableInputD, value } else &.{ matrixD, matrixA, matrixB, idesc, enableInputD } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.mma.ws.sp`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_mma_ws_sp(ctx: *mlir.Context, matrixD: *const mlir.Value, matrixA: *const mlir.Value, matrixB: *const mlir.Value, idesc: *const mlir.Value, enableInputD: *const mlir.Value, sparseMetadata: *const mlir.Value, zeroColMask: ?*const mlir.Value, kind: *const mlir.Attribute, collectorBBuffer: ?*const mlir.Attribute, collectorOp: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (collectorBBuffer) |value| attributes.appendAssumeCapacity(.named(ctx, "collectorBBuffer", value));
    if (collectorOp) |value| attributes.appendAssumeCapacity(.named(ctx, "collectorOp", value));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.mma.ws.sp", .{
        .operands = .{ .flat = if (zeroColMask) |value| &.{ matrixD, matrixA, matrixB, idesc, enableInputD, sparseMetadata, value } else &.{ matrixD, matrixA, matrixB, idesc, enableInputD, sparseMetadata } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.mma_smem_desc`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_mma_smem_desc(ctx: *mlir.Context, startAddr: *const mlir.Value, leadingDimOffset: *const mlir.Value, strideDimOffset: *const mlir.Value, baseOffset: *const mlir.Value, leadingDimMode: *const mlir.Value, swizzleMode: *const mlir.Value, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.tcgen05.mma_smem_desc", .{
        .operands = .{ .flat = &.{ startAddr, leadingDimOffset, strideDimOffset, baseOffset, leadingDimMode, swizzleMode } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.tcgen05.relinquish_alloc_permit`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_relinquish_alloc_permit(ctx: *mlir.Context, group: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (group) |value| attributes.appendAssumeCapacity(.named(ctx, "group", value));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.relinquish_alloc_permit", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.shift`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_shift(ctx: *mlir.Context, taddr: *const mlir.Value, group: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (group) |value| attributes.appendAssumeCapacity(.named(ctx, "group", value));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.shift", .{
        .operands = .{ .flat = &.{taddr} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.st`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_st(ctx: *mlir.Context, tmemAddr: *const mlir.Value, val: *const mlir.Value, offset: ?*const mlir.Value, unpack: ?*const mlir.Attribute, shape: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (unpack) |value| attributes.appendAssumeCapacity(.named(ctx, "unpack", value));
    attributes.appendAssumeCapacity(.named(ctx, "shape", shape));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.st", .{
        .operands = .{ .flat = if (offset) |value| &.{ tmemAddr, val, value } else &.{ tmemAddr, val } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.wait`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_wait(ctx: *mlir.Context, kind: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.tcgen05.wait", .{
        .attributes = &.{
            .named(ctx, "kind", kind),
        },
        .location = location,
    });
}

/// `nvvm.tcgen05_mma_smem_desc_v2`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_mma_smem_desc_v2(ctx: *mlir.Context, startAddress: *const mlir.Value, leadingDimOffset: *const mlir.Value, strideDimOffset: *const mlir.Value, descriptorVersion: *const mlir.Value, baseOffset: *const mlir.Value, leadingDimMode: *const mlir.Value, kSegmentOffset: *const mlir.Value, swizzleType: *const mlir.Value, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.tcgen05_mma_smem_desc_v2", .{
        .operands = .{ .flat = &.{ startAddress, leadingDimOffset, strideDimOffset, descriptorVersion, baseOffset, leadingDimMode, kSegmentOffset, swizzleType } },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `nvvm.tensormap.cp_fenceproxy`. Result types are explicit; attributes use mlir.Attribute.
pub fn tensormap_cp_fenceproxy(ctx: *mlir.Context, dst: *const mlir.Value, src: *const mlir.Value, size: *const mlir.Value, scope: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.tensormap.cp_fenceproxy", .{
        .operands = .{ .flat = &.{ dst, src, size } },
        .attributes = &.{
            .named(ctx, "scope", scope),
        },
        .location = location,
    });
}

/// `nvvm.tensormap.replace`. Result types are explicit; attributes use mlir.Attribute.
pub fn tensormap_replace(ctx: *mlir.Context, addr: *const mlir.Value, new_value: ?*const mlir.Value, field: *const mlir.Attribute, ord: ?*const mlir.Attribute, new_value_attr: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "field", field));
    if (ord) |value| attributes.appendAssumeCapacity(.named(ctx, "ord", value));
    if (new_value_attr) |value| attributes.appendAssumeCapacity(.named(ctx, "new_value_attr", value));
    return mlir.Operation.make(ctx, "nvvm.tensormap.replace", .{
        .operands = .{ .flat = if (new_value) |value| &.{ addr, value } else &.{addr} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.trace.mark`. Result types are explicit; attributes use mlir.Attribute.
pub fn trace_mark(ctx: *mlir.Context, payload: ?*const mlir.Value, predicate: ?*const mlir.Value, eventType: *const mlir.Attribute, domain: *const mlir.Attribute, event: *const mlir.Attribute, payloadDescriptor: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "eventType", eventType));
    attributes.appendAssumeCapacity(.named(ctx, "domain", domain));
    attributes.appendAssumeCapacity(.named(ctx, "event", event));
    if (payloadDescriptor) |value| attributes.appendAssumeCapacity(.named(ctx, "payloadDescriptor", value));
    return mlir.Operation.make(ctx, "nvvm.trace.mark", .{
        .operands = .{ .variadic = &.{
            if (payload) |value| &.{value} else &.{},
            if (predicate) |value| &.{value} else &.{},
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.vote.sync`. Result types are explicit; attributes use mlir.Attribute.
pub fn vote_sync(ctx: *mlir.Context, mask: *const mlir.Value, pred: *const mlir.Value, res_type: *const mlir.Type, kind: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.vote.sync", .{
        .operands = .{ .flat = &.{ mask, pred } },
        .results = .{ .flat = &.{res_type} },
        .attributes = &.{
            .named(ctx, "kind", kind),
        },
        .location = location,
    });
}

/// `nvvm.wgmma.commit.group.sync.aligned`. Result types are explicit; attributes use mlir.Attribute.
pub fn wgmma_commit_group_sync_aligned(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.wgmma.commit.group.sync.aligned", .{
        .location = location,
    });
}

/// `nvvm.wgmma.fence.aligned`. Result types are explicit; attributes use mlir.Attribute.
pub fn wgmma_fence_aligned(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.wgmma.fence.aligned", .{
        .location = location,
    });
}

/// `nvvm.wgmma.mma_async`. Result types are explicit; attributes use mlir.Attribute.
pub fn wgmma_mma_async(ctx: *mlir.Context, inouts: *const mlir.Value, descriptorA: *const mlir.Value, descriptorB: *const mlir.Value, results__type: *const mlir.Type, shape: *const mlir.Attribute, typeA: *const mlir.Attribute, typeB: *const mlir.Attribute, typeD: *const mlir.Attribute, scaleD: *const mlir.Attribute, scaleA: *const mlir.Attribute, scaleB: *const mlir.Attribute, layoutA: *const mlir.Attribute, layoutB: *const mlir.Attribute, satfinite: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 10) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "shape", shape));
    attributes.appendAssumeCapacity(.named(ctx, "typeA", typeA));
    attributes.appendAssumeCapacity(.named(ctx, "typeB", typeB));
    attributes.appendAssumeCapacity(.named(ctx, "typeD", typeD));
    attributes.appendAssumeCapacity(.named(ctx, "scaleD", scaleD));
    attributes.appendAssumeCapacity(.named(ctx, "scaleA", scaleA));
    attributes.appendAssumeCapacity(.named(ctx, "scaleB", scaleB));
    attributes.appendAssumeCapacity(.named(ctx, "layoutA", layoutA));
    attributes.appendAssumeCapacity(.named(ctx, "layoutB", layoutB));
    if (satfinite) |value| attributes.appendAssumeCapacity(.named(ctx, "satfinite", value));
    return mlir.Operation.make(ctx, "nvvm.wgmma.mma_async", .{
        .operands = .{ .flat = &.{ inouts, descriptorA, descriptorB } },
        .results = .{ .flat = &.{results__type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.wgmma.wait.group.sync.aligned`. Result types are explicit; attributes use mlir.Attribute.
pub fn wgmma_wait_group_sync_aligned(ctx: *mlir.Context, group: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.wgmma.wait.group.sync.aligned", .{
        .attributes = &.{
            .named(ctx, "group", group),
        },
        .location = location,
    });
}

/// `nvvm.wmma.load`. Result types are explicit; attributes use mlir.Attribute.
pub fn wmma_load(ctx: *mlir.Context, ptr: *const mlir.Value, stride: *const mlir.Value, res_type: *const mlir.Type, m: *const mlir.Attribute, n: *const mlir.Attribute, k: *const mlir.Attribute, layout: *const mlir.Attribute, eltype: *const mlir.Attribute, frag: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.wmma.load", .{
        .operands = .{ .flat = &.{ ptr, stride } },
        .results = .{ .flat = &.{res_type} },
        .attributes = &.{
            .named(ctx, "m", m),
            .named(ctx, "n", n),
            .named(ctx, "k", k),
            .named(ctx, "layout", layout),
            .named(ctx, "eltype", eltype),
            .named(ctx, "frag", frag),
        },
        .location = location,
    });
}

/// `nvvm.wmma.mma`. Result types are explicit; attributes use mlir.Attribute.
pub fn wmma_mma(ctx: *mlir.Context, args: []const *const mlir.Value, res_type: *const mlir.Type, m: *const mlir.Attribute, n: *const mlir.Attribute, k: *const mlir.Attribute, layoutA: *const mlir.Attribute, layoutB: *const mlir.Attribute, eltypeA: *const mlir.Attribute, eltypeB: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.wmma.mma", .{
        .operands = .{ .flat = args },
        .results = .{ .flat = &.{res_type} },
        .attributes = &.{
            .named(ctx, "m", m),
            .named(ctx, "n", n),
            .named(ctx, "k", k),
            .named(ctx, "layoutA", layoutA),
            .named(ctx, "layoutB", layoutB),
            .named(ctx, "eltypeA", eltypeA),
            .named(ctx, "eltypeB", eltypeB),
        },
        .location = location,
    });
}

/// `nvvm.wmma.store`. Result types are explicit; attributes use mlir.Attribute.
pub fn wmma_store(ctx: *mlir.Context, ptr: *const mlir.Value, args: []const *const mlir.Value, stride: *const mlir.Value, m: *const mlir.Attribute, n: *const mlir.Attribute, k: *const mlir.Attribute, layout: *const mlir.Attribute, eltype: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 256) = .empty;
    operands.appendAssumeCapacity(ptr);
    operands.appendSliceAssumeCapacity(args);
    operands.appendAssumeCapacity(stride);
    return mlir.Operation.make(ctx, "nvvm.wmma.store", .{
        .operands = .{ .flat = operands.constSlice() },
        .attributes = &.{
            .named(ctx, "m", m),
            .named(ctx, "n", n),
            .named(ctx, "k", k),
            .named(ctx, "layout", layout),
            .named(ctx, "eltype", eltype),
        },
        .location = location,
    });
}

test "every enum attribute prints as the compiler does" {
    @setEvalBranchQuota(1_000_000);
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    mlir.DialectHandle.fromString(dialect_handle).insertDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    inline for (.{
        .{ AtomicOpKindAttr, "#nvvm<atomic_op {s}>" },
        .{ BarrierReductionAttr, "#nvvm.reduction<{s}>" },
        .{ BarrierReduxKindAttr, "#nvvm.barrier_redux_kind<{s}>" },
        .{ BlockScaleFormatAttr, "#nvvm.block_scale_format<{s}>" },
        .{ CTAGroupKindAttr, "#nvvm.cta_group<{s}>" },
        .{ CVTPackFloatKindAttr, "#nvvm.packfloat_type<{s}>" },
        .{ CacheEvictionPriorityAttr, "#nvvm<cache_eviction_priority {s}>" },
        .{ ClusterLaunchControlQueryTypeAttr, "#nvvm<cluster_launch_control_query_type {s}>" },
        .{ CompareOpKindAttr, "#nvvm.op<{s}>" },
        .{ ConvertFP4TypeAttr, "#nvvm.convert_fp4_type<{s}>" },
        .{ ConvertFP8TypeAttr, "#nvvm.convert_fp8_type<{s}>" },
        .{ ConvertScaleKindAttr, "#nvvm.convert_scale_kind<{s}>" },
        .{ DotAccumulateTypeAttr, "#nvvm.dot_accumulate_type<{s}>" },
        .{ EvictKindAttr, "#nvvm.evict_kind<{s}>" },
        .{ FPRoundingModeAttr, "#nvvm.fp_rnd_mode<{s}>" },
        .{ GridDepActionKindAttr, "#nvvm<grid_dep_action {s}>" },
        .{ IntegerRoundingModeAttr, "#nvvm.int_rnd_mode<{s}>" },
        .{ L2PrefetchSizeAttr, "#nvvm.l2_prefetch<{s}>" },
        .{ LdStMatrixEltTypeAttr, "#nvvm.ld_st_matrix_elt_type<{s}>" },
        .{ LoadCacheModifierExtKindAttr, "#nvvm.load_cache_modifier_ext<{s}>" },
        .{ LoadCacheModifierKindAttr, "#nvvm<load_cache_modifier {s}>" },
        .{ LoadShapeAttr, "#nvvm.load_shape<{s}>" },
        .{ LoadSrcFormatAttr, "#nvvm.load_src_format<{s}>" },
        .{ MBarrierLayoutAttr, "#nvvm.mbarrier_layout<{s}>" },
        .{ MBarrierPhaseAttr, "#nvvm.mbarrier_phase<{s}>" },
        .{ MBarrierScopeKindAttr, "#nvvm.mbar_scope<{s}>" },
        .{ MBarrierSpaceKindAttr, "#nvvm.mbar_space<{s}>" },
        .{ MBarrierTxnKindAttr, "#nvvm.mbar_txn_kind<{s}>" },
        .{ MBarrierWaitKindAttr, "#nvvm.mbar_wait<{s}>" },
        .{ MMAB1OpAttr, "#nvvm.mma_b1op<{s}>" },
        .{ MMABlockScaleKindAttr, "#nvvm.block_scale_kind<{s}>" },
        .{ MMACtaCountAttr, "#nvvm.mma_cta_count<{s}>" },
        .{ MMAFragAttr, "#nvvm.mma_frag<{s}>" },
        .{ MMAIntOverflowAttr, "#nvvm.mma_int_overflow<{s}>" },
        .{ MMAKindAttr, "#nvvm.mma_kind<{s}>" },
        .{ MMALayoutAttr, "#nvvm.mma_layout<{s}>" },
        .{ MMATypesAttr, "#nvvm.mma_type<{s}>" },
        .{ MatchSyncKindAttr, "#nvvm<match_sync_kind {s}>" },
        .{ MemOrderKindAttr, "#nvvm.mem_order<{s}>" },
        .{ MemScopeKindAttr, "#nvvm.mem_scope<{s}>" },
        .{ MulModeAttr, "#nvvm<mul_mode {s}>" },
        .{ NVVMMemorySpaceAttr, "#nvvm.memory_space<{s}>" },
        .{ PermuteModeAttr, "#nvvm.permute_mode<{s}>" },
        .{ PrefetchCacheLevelAttr, "#nvvm<prefetch_cache_level {s}>" },
        .{ ProxyKindAttr, "#nvvm.proxy_kind<{s}>" },
        .{ ReductionKindAttr, "#nvvm<reduction_kind {s}>" },
        .{ ReductionOpAttr, "#nvvm.red_op<{s}>" },
        .{ ReductionTypeAttr, "#nvvm.red_type<{s}>" },
        .{ SPCompressElemSizeAttr, "#nvvm.spcomp_elem_size<{s}>" },
        .{ SPCompressFactorTypeAttr, "#nvvm.spcomp_factor<{s}>" },
        .{ SPCompressIndexSizeAttr, "#nvvm.spcomp_index_size<{s}>" },
        .{ SPCompressOpKindAttr, "#nvvm.spcomp_op_kind<{s}>" },
        .{ SPCompressRepFactorAttr, "#nvvm.spcomp_rep_factor<{s}>" },
        .{ SPDecompressElemSizeAttr, "#nvvm.spdecomp_elem_size<{s}>" },
        .{ SPDecompressFactorTypeAttr, "#nvvm.spdecomp_factor<{s}>" },
        .{ SPDecompressIndexSizeAttr, "#nvvm.spdecomp_index_size<{s}>" },
        .{ SPDecompressRepFactorAttr, "#nvvm.spdecomp_rep_factor<{s}>" },
        .{ SaturationModeAttr, "#nvvm.sat_mode<{s}>" },
        .{ SaturationModeKindAttr, "#nvvm.sat<{s}>" },
        .{ ScaleVecSizeAttr, "#nvvm.scale_vec_size<{s}>" },
        .{ SetMaxRegisterActionAttr, "#nvvm<action {s}>" },
        .{ SharedSpaceAttr, "#nvvm.shared_space<{s}>" },
        .{ ShflKindAttr, "#nvvm<shfl_kind {s}>" },
        .{ SparsityFormatAttr, "#nvvm.sparsity_format<{s}>" },
        .{ StateSpaceAttr, "#nvvm.state_space<{s}>" },
        .{ StoreCacheModifierKindAttr, "#nvvm.store_cache_modifier<{s}>" },
        .{ StoreShapeAttr, "#nvvm.store_shape<{s}>" },
        .{ TCBarParamAttr, "#nvvm.TCBarParam<{s}>" },
        .{ TMALoadModeAttr, "#nvvm.tma_load_mode<{s}>" },
        .{ TMAReduxKindAttr, "#nvvm.tma_redux_kind<{s}>" },
        .{ TMAStoreModeAttr, "#nvvm.tma_store_mode<{s}>" },
        .{ Tcgen05CpMulticastAttr, "#nvvm.tcgen05_cp_multicast<{s}>" },
        .{ Tcgen05CpShapeAttr, "#nvvm.tcgen05_cp_shape<{s}>" },
        .{ Tcgen05CpSrcFormatAttr, "#nvvm.tcgen05_cp_src_fmt<{s}>" },
        .{ Tcgen05FenceKindAttr, "#nvvm.tcgen05_fence<{s}>" },
        .{ Tcgen05LdStShapeAttr, "#nvvm.tcgen05_ldst_shape<{s}>" },
        .{ Tcgen05MMABlockScaleAttr, "#nvvm.tcgen05_mma_block_scale<{s}>" },
        .{ Tcgen05MMACollectorBBufferAttr, "#nvvm.tcgen05_mma_collectorb<{s}>" },
        .{ Tcgen05MMACollectorOpAttr, "#nvvm.tcgen05_mma_collectorop<{s}>" },
        .{ Tcgen05MMAKindAttr, "#nvvm.tcgen05_mma_kind<{s}>" },
        .{ Tcgen05WaitKindAttr, "#nvvm.tcgen05_wait<{s}>" },
        .{ TensormapElemtypeAttr, "#nvvm.tensormap_elemtype<{s}>" },
        .{ TensormapFieldAttr, "#nvvm<tensormap_field {s}>" },
        .{ TensormapFillModeAttr, "#nvvm.tensormap_fill_mode<{s}>" },
        .{ TensormapInterleaveLayoutAttr, "#nvvm.tensormap_interleave_layout<{s}>" },
        .{ TensormapSwizzleAtomicityAttr, "#nvvm.tensormap_swizzle_atomicity<{s}>" },
        .{ TensormapSwizzleModeAttr, "#nvvm.tensormap_swizzle_mode<{s}>" },
        .{ TmemLayoutAttr, "#nvvm.TmemLayout<{s}>" },
        .{ ValidatePatternAttr, "#nvvm.validate_pattern<{s}>" },
        .{ VoteSyncKindAttr, "#nvvm<vote_sync_kind {s}>" },
        .{ WGMMAScaleInAttr, "#nvvm.wgmma_scale_in<{s}>" },
        .{ WGMMAScaleOutAttr, "#nvvm.wgmma_scale_out<{s}>" },
        .{ WGMMATypesAttr, "#nvvm.wgmma_type<{s}>" },
    }) |example| {
        const T = example[0];
        inline for (@typeInfo(@FieldType(T.InitArgs, "value")).@"enum".fields) |case| {
            const attr = try T.get(ctx, .{ .value = @enumFromInt(case.value) });
            try std.testing.expectEqual(case.value, @intFromEnum(attr.getValue()));
            var buf: [256]u8 = undefined;
            var w: std.Io.Writer = .fixed(&buf);
            try w.print("{f}", .{attr});
            try std.testing.expectEqualStrings(std.fmt.comptimePrint(example[1], .{case.name}), w.buffered());
        }
    }
}

/// Every operation bound here.
pub const operation_names: []const []const u8 = &.{
    "nvvm.add.packed.bf16x2.f32x2.f32x2",
    "nvvm.add.packed.f16x2.f32x2.f32x2",
    "nvvm.add.packed.f32x2",
    "nvvm.add.packed.f32x2.bf16x2.f32x2",
    "nvvm.add.packed.f32x2.f16x2.f32x2",
    "nvvm.addf",
    "nvvm.applypriority.async.bulk",
    "nvvm.applypriority.async.bulk.tensor",
    "nvvm.applypriority.async.bulk.tensor.override",
    "nvvm.atomicrmw",
    "nvvm.bar.warp.sync",
    "nvvm.barrier",
    "nvvm.barrier.arrive",
    "nvvm.barrier.cta.arrive",
    "nvvm.barrier.cta.red",
    "nvvm.barrier.cta.sync",
    "nvvm.barrier0",
    "nvvm.breakpoint",
    "nvvm.clmad",
    "nvvm.cluster.arrive",
    "nvvm.cluster.arrive.relaxed",
    "nvvm.cluster.wait",
    "nvvm.clusterlaunchcontrol.query.cancel",
    "nvvm.clusterlaunchcontrol.try.cancel",
    "nvvm.compare.and.set",
    "nvvm.convert.and.pack.integer",
    "nvvm.convert.bf16x2.to.f4x2",
    "nvvm.convert.bf16x2.to.f6x2",
    "nvvm.convert.bf16x2.to.f8x2",
    "nvvm.convert.bf16x2.to.s2f6x2",
    "nvvm.convert.f16x2.to.f4x2",
    "nvvm.convert.f16x2.to.f6x2",
    "nvvm.convert.f16x2.to.f8x2",
    "nvvm.convert.f32x2.to.bf16x2",
    "nvvm.convert.f32x2.to.f16x2",
    "nvvm.convert.f32x2.to.f4x2",
    "nvvm.convert.f32x2.to.f6x2",
    "nvvm.convert.f32x2.to.f8x2",
    "nvvm.convert.f32x2.to.s2f6x2",
    "nvvm.convert.f32x4.to.f4x4",
    "nvvm.convert.f32x4.to.f6x4",
    "nvvm.convert.f32x4.to.f8x4",
    "nvvm.convert.f4x2.to.f16x2",
    "nvvm.convert.f6x2.to.f16x2",
    "nvvm.convert.f8x2.to.bf16x2",
    "nvvm.convert.f8x2.to.f16x2",
    "nvvm.convert.float.to.integer",
    "nvvm.convert.float.to.tf32",
    "nvvm.convert.s2f6x2.to.bf16x2",
    "nvvm.cos",
    "nvvm.cp.async.bulk.commit.group",
    "nvvm.cp.async.bulk.global.shared.cta",
    "nvvm.cp.async.bulk.prefetch",
    "nvvm.cp.async.bulk.shared.cluster.global",
    "nvvm.cp.async.bulk.shared.cluster.shared.cta",
    "nvvm.cp.async.bulk.tensor.global.shared.cta",
    "nvvm.cp.async.bulk.tensor.global.shared.cta.override",
    "nvvm.cp.async.bulk.tensor.prefetch",
    "nvvm.cp.async.bulk.tensor.prefetch.override",
    "nvvm.cp.async.bulk.tensor.reduce",
    "nvvm.cp.async.bulk.tensor.reduce.override",
    "nvvm.cp.async.bulk.tensor.shared.cluster.global",
    "nvvm.cp.async.bulk.tensor.shared.cluster.global.override",
    "nvvm.cp.async.bulk.wait_group",
    "nvvm.cp.async.commit.group",
    "nvvm.cp.async.mbarrier.arrive",
    "nvvm.cp.async.shared.global",
    "nvvm.cp.async.wait.group",
    "nvvm.cvt.packfloat",
    "nvvm.cvt.packfloat.f32",
    "nvvm.cvt.to.f4x8.packed",
    "nvvm.dot.accumulate.2way",
    "nvvm.dot.accumulate.4way",
    "nvvm.elect.sync",
    "nvvm.ex2",
    "nvvm.exit",
    "nvvm.fabs",
    "nvvm.fence.mbarrier.init",
    "nvvm.fence.proxy",
    "nvvm.fence.proxy.acquire",
    "nvvm.fence.proxy.release",
    "nvvm.fence.proxy.sync_restrict",
    "nvvm.fence.sc.cluster",
    "nvvm.fence.sync_restrict",
    "nvvm.fma",
    "nvvm.fma.packed.f32x2",
    "nvvm.fma.packed.f32x2.bf16x2.f32x2.f32x2",
    "nvvm.fma.packed.f32x2.f16x2.f32x2.f32x2",
    "nvvm.fmax",
    "nvvm.fmin",
    "nvvm.griddepcontrol",
    "nvvm.inline_ptx",
    "nvvm.ldmatrix",
    "nvvm.load.ext",
    "nvvm.log2",
    "nvvm.mapa",
    "nvvm.match.sync",
    "nvvm.mbarrier.arrive",
    "nvvm.mbarrier.arrive.expect_tx",
    "nvvm.mbarrier.arrive.nocomplete",
    "nvvm.mbarrier.arrive_drop",
    "nvvm.mbarrier.arrive_drop.expect_tx",
    "nvvm.mbarrier.arrive_drop.nocomplete",
    "nvvm.mbarrier.check.layout",
    "nvvm.mbarrier.complete_tx",
    "nvvm.mbarrier.expect_tx",
    "nvvm.mbarrier.init",
    "nvvm.mbarrier.inval",
    "nvvm.mbarrier.test.wait",
    "nvvm.mbarrier.try_wait",
    "nvvm.mbarrier.try_wait.parity",
    "nvvm.mbarrier.try_wait.parity.timelimit",
    "nvvm.mbarrier.try_wait.timelimit",
    "nvvm.mbarrier.txn",
    "nvvm.mbarrier.txn.cta",
    "nvvm.mbarrier.wait",
    "nvvm.mbarrier.wait.parity",
    "nvvm.memory.barrier",
    "nvvm.mma.block_scale",
    "nvvm.mma.block_scale.internal",
    "nvvm.mma.sp.block_scale",
    "nvvm.mma.sp.block_scale.internal",
    "nvvm.mma.sp.sync",
    "nvvm.mma.sync",
    "nvvm.mma_smem_desc",
    "nvvm.movmatrix",
    "nvvm.mul",
    "nvvm.mul.packed.bf16x2.bf16x2.f16x2",
    "nvvm.mul.packed.bf16x2.f32x2.f32x2",
    "nvvm.mul.packed.f16x2.f16x2.bf16x2",
    "nvvm.mul.packed.f16x2.f32x2.f32x2",
    "nvvm.mul.packed.f32x2",
    "nvvm.multimem.ld.reduce",
    "nvvm.multimem.red",
    "nvvm.multimem.st",
    "nvvm.nanosleep",
    "nvvm.pmevent",
    "nvvm.prefetch",
    "nvvm.prmt",
    "nvvm.rcp.approx.ftz.f",
    "nvvm.read.ptx.sreg.aggr.smem.size",
    "nvvm.read.ptx.sreg.clock",
    "nvvm.read.ptx.sreg.clock64",
    "nvvm.read.ptx.sreg.cluster.ctaid.x",
    "nvvm.read.ptx.sreg.cluster.ctaid.y",
    "nvvm.read.ptx.sreg.cluster.ctaid.z",
    "nvvm.read.ptx.sreg.cluster.ctarank",
    "nvvm.read.ptx.sreg.cluster.nctaid.x",
    "nvvm.read.ptx.sreg.cluster.nctaid.y",
    "nvvm.read.ptx.sreg.cluster.nctaid.z",
    "nvvm.read.ptx.sreg.cluster.nctarank",
    "nvvm.read.ptx.sreg.clusterid.x",
    "nvvm.read.ptx.sreg.clusterid.y",
    "nvvm.read.ptx.sreg.clusterid.z",
    "nvvm.read.ptx.sreg.ctaid.x",
    "nvvm.read.ptx.sreg.ctaid.y",
    "nvvm.read.ptx.sreg.ctaid.z",
    "nvvm.read.ptx.sreg.dynamic.smem.size",
    "nvvm.read.ptx.sreg.envreg0",
    "nvvm.read.ptx.sreg.envreg1",
    "nvvm.read.ptx.sreg.envreg10",
    "nvvm.read.ptx.sreg.envreg11",
    "nvvm.read.ptx.sreg.envreg12",
    "nvvm.read.ptx.sreg.envreg13",
    "nvvm.read.ptx.sreg.envreg14",
    "nvvm.read.ptx.sreg.envreg15",
    "nvvm.read.ptx.sreg.envreg16",
    "nvvm.read.ptx.sreg.envreg17",
    "nvvm.read.ptx.sreg.envreg18",
    "nvvm.read.ptx.sreg.envreg19",
    "nvvm.read.ptx.sreg.envreg2",
    "nvvm.read.ptx.sreg.envreg20",
    "nvvm.read.ptx.sreg.envreg21",
    "nvvm.read.ptx.sreg.envreg22",
    "nvvm.read.ptx.sreg.envreg23",
    "nvvm.read.ptx.sreg.envreg24",
    "nvvm.read.ptx.sreg.envreg25",
    "nvvm.read.ptx.sreg.envreg26",
    "nvvm.read.ptx.sreg.envreg27",
    "nvvm.read.ptx.sreg.envreg28",
    "nvvm.read.ptx.sreg.envreg29",
    "nvvm.read.ptx.sreg.envreg3",
    "nvvm.read.ptx.sreg.envreg30",
    "nvvm.read.ptx.sreg.envreg31",
    "nvvm.read.ptx.sreg.envreg4",
    "nvvm.read.ptx.sreg.envreg5",
    "nvvm.read.ptx.sreg.envreg6",
    "nvvm.read.ptx.sreg.envreg7",
    "nvvm.read.ptx.sreg.envreg8",
    "nvvm.read.ptx.sreg.envreg9",
    "nvvm.read.ptx.sreg.globaltimer",
    "nvvm.read.ptx.sreg.globaltimer.lo",
    "nvvm.read.ptx.sreg.gridid",
    "nvvm.read.ptx.sreg.laneid",
    "nvvm.read.ptx.sreg.lanemask.eq",
    "nvvm.read.ptx.sreg.lanemask.ge",
    "nvvm.read.ptx.sreg.lanemask.gt",
    "nvvm.read.ptx.sreg.lanemask.le",
    "nvvm.read.ptx.sreg.lanemask.lt",
    "nvvm.read.ptx.sreg.nclusterid.x",
    "nvvm.read.ptx.sreg.nclusterid.y",
    "nvvm.read.ptx.sreg.nclusterid.z",
    "nvvm.read.ptx.sreg.nctaid.x",
    "nvvm.read.ptx.sreg.nctaid.y",
    "nvvm.read.ptx.sreg.nctaid.z",
    "nvvm.read.ptx.sreg.nsmid",
    "nvvm.read.ptx.sreg.ntid.x",
    "nvvm.read.ptx.sreg.ntid.y",
    "nvvm.read.ptx.sreg.ntid.z",
    "nvvm.read.ptx.sreg.nwarpid",
    "nvvm.read.ptx.sreg.smid",
    "nvvm.read.ptx.sreg.tid.x",
    "nvvm.read.ptx.sreg.tid.y",
    "nvvm.read.ptx.sreg.tid.z",
    "nvvm.read.ptx.sreg.total.smem.size",
    "nvvm.read.ptx.sreg.warpid",
    "nvvm.read.ptx.sreg.warpsize",
    "nvvm.red",
    "nvvm.redux.sync",
    "nvvm.rsqrt",
    "nvvm.setmaxregister",
    "nvvm.shfl.sync",
    "nvvm.sin",
    "nvvm.spcompress",
    "nvvm.spdecompress",
    "nvvm.st.bulk",
    "nvvm.stmatrix",
    "nvvm.store.async",
    "nvvm.store.ext",
    "nvvm.sub.packed.bf16x2.f32x2.f32x2",
    "nvvm.sub.packed.f16x2.f32x2.f32x2",
    "nvvm.sub.packed.f32x2",
    "nvvm.sub.packed.f32x2.bf16x2.f32x2",
    "nvvm.sub.packed.f32x2.f16x2.f32x2",
    "nvvm.subf",
    "nvvm.tcgen05.alloc",
    "nvvm.tcgen05.commit",
    "nvvm.tcgen05.cp",
    "nvvm.tcgen05.dealloc",
    "nvvm.tcgen05.fence",
    "nvvm.tcgen05.ld",
    "nvvm.tcgen05.ld.red",
    "nvvm.tcgen05.ld.red.spcompress",
    "nvvm.tcgen05.ld.spcompress",
    "nvvm.tcgen05.mma",
    "nvvm.tcgen05.mma.block_scale",
    "nvvm.tcgen05.mma.block_scale.decompress_b",
    "nvvm.tcgen05.mma.decompress_b",
    "nvvm.tcgen05.mma.sp",
    "nvvm.tcgen05.mma.sp.block_scale",
    "nvvm.tcgen05.mma.ws",
    "nvvm.tcgen05.mma.ws.sp",
    "nvvm.tcgen05.mma_smem_desc",
    "nvvm.tcgen05.relinquish_alloc_permit",
    "nvvm.tcgen05.shift",
    "nvvm.tcgen05.st",
    "nvvm.tcgen05.wait",
    "nvvm.tcgen05_mma_smem_desc_v2",
    "nvvm.tensormap.cp_fenceproxy",
    "nvvm.tensormap.replace",
    "nvvm.trace.mark",
    "nvvm.vote.sync",
    "nvvm.wgmma.commit.group.sync.aligned",
    "nvvm.wgmma.fence.aligned",
    "nvvm.wgmma.mma_async",
    "nvvm.wgmma.wait.group.sync.aligned",
    "nvvm.wmma.load",
    "nvvm.wmma.mma",
    "nvvm.wmma.store",
};
