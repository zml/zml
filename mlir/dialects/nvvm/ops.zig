//! Zig bindings of NVIDIA's NVVM dialect of the CuTe DSL (nvidia-cutlass-dsl 4.8.0), generated
//! from the dialect's .td files; do not edit.

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
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "atomic_op";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: AtomicOpKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMAtomicOpKindAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) AtomicOpKind {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMAtomicOpKindAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.reduction`: NVVM barrier reduction operation
pub const BarrierReductionAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMBarrierReduction;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "reduction";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: BarrierReduction,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMBarrierReductionAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) BarrierReduction {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMBarrierReductionAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.barrier_redux_kind`: NVVM barrier redux kind
pub const BarrierReduxKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMBarrierReduxKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "barrier_redux_kind";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: BarrierReduxKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMBarrierReduxKindAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) BarrierReduxKind {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMBarrierReduxKindAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.block_scale_format`: MMA Block Scale Format
pub const BlockScaleFormatAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMBlockScaleFormat;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "block_scale_format";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: BlockScaleFormat,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMBlockScaleFormatAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) BlockScaleFormat {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMBlockScaleFormatAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.cta_group`: NVVM CTA group kind
pub const CTAGroupKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMCTAGroupKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "cta_group";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: CTAGroupKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMCTAGroupKindAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) CTAGroupKind {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMCTAGroupKindAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.packfloat_type`: NVVM CVT Pack Float kind
pub const CVTPackFloatKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMCVTPackFloatKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "packfloat_type";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: CVTPackFloatKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMCVTPackFloatKindAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) CVTPackFloatKind {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMCVTPackFloatKindAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.cache_eviction_priority`: NVVM Cache Eviction Priority
pub const CacheEvictionPriorityAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMCacheEvictionPriority;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "cache_eviction_priority";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: CacheEvictionPriority,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMCacheEvictionPriorityAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) CacheEvictionPriority {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMCacheEvictionPriorityAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.cluster_launch_control_query_type`: NVVM ClusterLaunchControlQueryType
pub const ClusterLaunchControlQueryTypeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMClusterLaunchControlQueryType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "cluster_launch_control_query_type";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: ClusterLaunchControlQueryType,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMClusterLaunchControlQueryTypeAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) ClusterLaunchControlQueryType {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMClusterLaunchControlQueryTypeAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.op`: Comparison operator encoding
pub const CompareOpKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMCompareOpKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "op";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: CompareOpKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMCompareOpKindAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) CompareOpKind {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMCompareOpKindAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.convert_fp4_type`: NVVM ConvertFP4Type kind
pub const ConvertFP4TypeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMConvertFP4Type;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "convert_fp4_type";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: ConvertFP4Type,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMConvertFP4TypeAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) ConvertFP4Type {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMConvertFP4TypeAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.convert_fp8_type`: NVVM ConvertFP8Type kind
pub const ConvertFP8TypeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMConvertFP8Type;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "convert_fp8_type";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: ConvertFP8Type,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMConvertFP8TypeAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) ConvertFP8Type {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMConvertFP8TypeAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.convert_scale_kind`: NVVM ConvertScale kind
pub const ConvertScaleKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMConvertScaleKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "convert_scale_kind";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: ConvertScaleKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMConvertScaleKindAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) ConvertScaleKind {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMConvertScaleKindAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.dot_accumulate_type`: NVVM DotAccumulateType
pub const DotAccumulateTypeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMDotAccumulateType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "dot_accumulate_type";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: DotAccumulateType,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMDotAccumulateTypeAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) DotAccumulateType {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMDotAccumulateTypeAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.evict_kind`: NVVM L2 Prefetch Size
pub const EvictKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMEvictKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "evict_kind";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: EvictKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMEvictKindAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) EvictKind {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMEvictKindAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.fp_rnd_mode`: NVVM FPRoundingMode kind
pub const FPRoundingModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMFPRoundingMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "fp_rnd_mode";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: FPRoundingMode,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMFPRoundingModeAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) FPRoundingMode {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMFPRoundingModeAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.grid_dep_action`: Action kind for grid dependency control
pub const GridDepActionKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMGridDepActionKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "grid_dep_action";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: GridDepActionKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMGridDepActionKindAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) GridDepActionKind {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMGridDepActionKindAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.int_rnd_mode`: NVVM IntegerRoundingMode kind
pub const IntegerRoundingModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMIntegerRoundingMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "int_rnd_mode";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: IntegerRoundingMode,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMIntegerRoundingModeAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) IntegerRoundingMode {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMIntegerRoundingModeAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.l2_prefetch`: NVVM L2 Prefetch Size
pub const L2PrefetchSizeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVML2PrefetchSize;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "l2_prefetch";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: L2PrefetchSize,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVML2PrefetchSizeAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) L2PrefetchSize {
        return @fromBackingInt(@intCast(c.mlirCuteNVVML2PrefetchSizeAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.ld_st_matrix_elt_type`: Element type for ldmatrix and stmatrix
pub const LdStMatrixEltTypeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMLdStMatrixEltType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "ld_st_matrix_elt_type";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: LdStMatrixEltType,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMLdStMatrixEltTypeAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) LdStMatrixEltType {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMLdStMatrixEltTypeAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.load_cache_modifier_ext`: NVVM load cache modifier kind(Ext)
pub const LoadCacheModifierExtKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMLoadCacheModifierExtKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "load_cache_modifier_ext";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: LoadCacheModifierExtKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMLoadCacheModifierExtKindAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) LoadCacheModifierExtKind {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMLoadCacheModifierExtKindAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.load_cache_modifier`: NVVM load cache modifier kind
pub const LoadCacheModifierKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMLoadCacheModifierKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "load_cache_modifier";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: LoadCacheModifierKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMLoadCacheModifierKindAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) LoadCacheModifierKind {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMLoadCacheModifierKindAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.load_shape`: shape attribute for ldmatrix
pub const LoadShapeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMLoadShape;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "load_shape";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: LoadShape,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMLoadShapeAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) LoadShape {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMLoadShapeAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.load_src_format`: source format for ldmatrix
pub const LoadSrcFormatAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMLoadSrcFormat;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "load_src_format";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: LoadSrcFormat,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMLoadSrcFormatAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) LoadSrcFormat {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMLoadSrcFormatAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.mbarrier_layout`: NVVM MBarrier Layout
pub const MBarrierLayoutAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMBarrierLayout;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "mbarrier_layout";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MBarrierLayout,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMBarrierLayoutAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) MBarrierLayout {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMMBarrierLayoutAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.mbarrier_phase`: NVVM mbarrier phase type
pub const MBarrierPhaseAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMBarrierPhase;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "mbarrier_phase";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MBarrierPhase,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMBarrierPhaseAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) MBarrierPhase {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMMBarrierPhaseAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.mbar_scope`: NVVM MBarrier scope kind
pub const MBarrierScopeKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMBarrierScopeKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "mbar_scope";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MBarrierScopeKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMBarrierScopeKindAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) MBarrierScopeKind {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMMBarrierScopeKindAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.mbar_space`: NVVM MBarrier space kind
pub const MBarrierSpaceKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMBarrierSpaceKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "mbar_space";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MBarrierSpaceKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMBarrierSpaceKindAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) MBarrierSpaceKind {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMMBarrierSpaceKindAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.mbar_txn_kind`: NVVM MBarrier Transaction kind
pub const MBarrierTxnKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMBarrierTxnKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "mbar_txn_kind";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MBarrierTxnKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMBarrierTxnKindAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) MBarrierTxnKind {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMMBarrierTxnKindAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.mbar_wait`: NVVM MBarrier wait kind
pub const MBarrierWaitKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMBarrierWaitKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "mbar_wait";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MBarrierWaitKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMBarrierWaitKindAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) MBarrierWaitKind {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMMBarrierWaitKindAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.mma_b1op`: MMA binary operations
pub const MMAB1OpAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMMAB1Op;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "mma_b1op";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MMAB1Op,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMMAB1OpAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) MMAB1Op {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMMMAB1OpAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.block_scale_kind`: Block Scale Kind
pub const MMABlockScaleKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMMABlockScaleKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "block_scale_kind";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MMABlockScaleKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMMABlockScaleKindAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) MMABlockScaleKind {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMMMABlockScaleKindAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.mma_cta_count`: MMA CTA count
pub const MMACtaCountAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMMACtaCount;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "mma_cta_count";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MMACtaCount,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMMACtaCountAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) MMACtaCount {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMMMACtaCountAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.mma_frag`: NVVM MMA frag type
pub const MMAFragAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMMAFrag;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "mma_frag";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MMAFrag,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMMAFragAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) MMAFrag {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMMMAFragAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.mma_int_overflow`: MMA overflow options
pub const MMAIntOverflowAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMMAIntOverflow;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "mma_int_overflow";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MMAIntOverflow,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMMAIntOverflowAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) MMAIntOverflow {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMMMAIntOverflowAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.mma_kind`: MMA operation kind
pub const MMAKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMMAKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "mma_kind";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MMAKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMMAKindAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) MMAKind {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMMMAKindAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.mma_layout`: NVVM MMA layout
pub const MMALayoutAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMMALayout;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "mma_layout";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MMALayout,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMMALayoutAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) MMALayout {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMMMALayoutAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.mma_type`: NVVM MMA types
pub const MMATypesAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMMATypes;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "mma_type";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MMATypes,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMMATypesAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) MMATypes {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMMMATypesAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.match_sync_kind`: NVVM match sync kind
pub const MatchSyncKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMatchSyncKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "match_sync_kind";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MatchSyncKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMatchSyncKindAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) MatchSyncKind {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMMatchSyncKindAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.mem_order`: NVVM Memory Ordering kind
pub const MemOrderKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMemOrderKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "mem_order";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MemOrderKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMemOrderKindAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) MemOrderKind {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMMemOrderKindAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.mem_scope`: NVVM Memory Scope kind
pub const MemScopeKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMemScopeKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "mem_scope";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MemScopeKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMemScopeKindAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) MemScopeKind {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMMemScopeKindAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.mul_mode`: multiply mode attribute
pub const MulModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMMulMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "mul_mode";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: MulMode,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMMulModeAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) MulMode {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMMulModeAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.memory_space`: NVVM Memory Space
pub const NVVMMemorySpaceAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMNVVMMemorySpace;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "memory_space";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: NVVMMemorySpace,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMNVVMMemorySpaceAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) NVVMMemorySpace {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMNVVMMemorySpaceAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.permute_mode`: NVVM permute mode
pub const PermuteModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMPermuteMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "permute_mode";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: PermuteMode,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMPermuteModeAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) PermuteMode {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMPermuteModeAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.prefetch_cache_level`: NVVM Prefetch Cache Level
pub const PrefetchCacheLevelAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMPrefetchCacheLevel;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "prefetch_cache_level";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: PrefetchCacheLevel,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMPrefetchCacheLevelAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) PrefetchCacheLevel {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMPrefetchCacheLevelAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.proxy_kind`: Proxy kind
pub const ProxyKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMProxyKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "proxy_kind";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: ProxyKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMProxyKindAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) ProxyKind {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMProxyKindAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.reduction_kind`: NVVM Reduction Kind attribute
pub const ReductionKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMReductionKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "reduction_kind";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: ReductionKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMReductionKindAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) ReductionKind {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMReductionKindAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.red_op`: Ops supported by red instruction
pub const ReductionOpAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMReductionOp;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "red_op";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: ReductionOp,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMReductionOpAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) ReductionOp {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMReductionOpAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.red_type`: types supported by red instruction
pub const ReductionTypeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMReductionType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "red_type";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: ReductionType,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMReductionTypeAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) ReductionType {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMReductionTypeAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.spcomp_elem_size`: Sparse tensor compression element size
pub const SPCompressElemSizeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMSPCompressElemSize;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "spcomp_elem_size";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: SPCompressElemSize,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMSPCompressElemSizeAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) SPCompressElemSize {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMSPCompressElemSizeAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.spcomp_factor`: Sparse tensor compression factor type
pub const SPCompressFactorTypeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMSPCompressFactorType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "spcomp_factor";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: SPCompressFactorType,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMSPCompressFactorTypeAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) SPCompressFactorType {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMSPCompressFactorTypeAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.spcomp_index_size`: Sparse tensor compression index size
pub const SPCompressIndexSizeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMSPCompressIndexSize;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "spcomp_index_size";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: SPCompressIndexSize,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMSPCompressIndexSizeAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) SPCompressIndexSize {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMSPCompressIndexSizeAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.spcomp_op_kind`: spcompress operation kind
pub const SPCompressOpKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMSPCompressOpKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "spcomp_op_kind";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: SPCompressOpKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMSPCompressOpKindAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) SPCompressOpKind {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMSPCompressOpKindAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.spcomp_rep_factor`: Sparse tensor compression repetition factor
pub const SPCompressRepFactorAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMSPCompressRepFactor;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "spcomp_rep_factor";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: SPCompressRepFactor,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMSPCompressRepFactorAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) SPCompressRepFactor {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMSPCompressRepFactorAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.spdecomp_elem_size`: Sparse tensor decompression element size
pub const SPDecompressElemSizeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMSPDecompressElemSize;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "spdecomp_elem_size";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: SPDecompressElemSize,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMSPDecompressElemSizeAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) SPDecompressElemSize {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMSPDecompressElemSizeAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.spdecomp_factor`: Sparse tensor decompression factor type
pub const SPDecompressFactorTypeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMSPDecompressFactorType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "spdecomp_factor";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: SPDecompressFactorType,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMSPDecompressFactorTypeAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) SPDecompressFactorType {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMSPDecompressFactorTypeAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.spdecomp_index_size`: Sparse tensor decompression index size
pub const SPDecompressIndexSizeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMSPDecompressIndexSize;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "spdecomp_index_size";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: SPDecompressIndexSize,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMSPDecompressIndexSizeAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) SPDecompressIndexSize {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMSPDecompressIndexSizeAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.spdecomp_rep_factor`: Sparse tensor decompression repetition factor
pub const SPDecompressRepFactorAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMSPDecompressRepFactor;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "spdecomp_rep_factor";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: SPDecompressRepFactor,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMSPDecompressRepFactorAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) SPDecompressRepFactor {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMSPDecompressRepFactorAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.sat_mode`: NVVM SaturationMode kind
pub const SaturationModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMSaturationMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "sat_mode";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: SaturationMode,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMSaturationModeAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) SaturationMode {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMSaturationModeAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.sat`: NVVM SaturationMode kind
pub const SaturationModeKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMSaturationModeKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "sat";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: SaturationModeKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMSaturationModeKindAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) SaturationModeKind {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMSaturationModeKindAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.scale_vec_size`: MMA Scale Vector Sizes
pub const ScaleVecSizeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMScaleVecSize;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "scale_vec_size";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: ScaleVecSize,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMScaleVecSizeAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) ScaleVecSize {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMScaleVecSizeAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.action`: NVVM set max register action
pub const SetMaxRegisterActionAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMSetMaxRegisterAction;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "action";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: SetMaxRegisterAction,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMSetMaxRegisterActionAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) SetMaxRegisterAction {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMSetMaxRegisterActionAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.shared_space`: Shared memory space
pub const SharedSpaceAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMSharedSpace;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "shared_space";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: SharedSpace,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMSharedSpaceAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) SharedSpace {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMSharedSpaceAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.shfl_kind`: NVVM shuffle kind
pub const ShflKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMShflKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "shfl_kind";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: ShflKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMShflKindAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) ShflKind {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMShflKindAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.sparsity_format`: MMA Sparsity Format
pub const SparsityFormatAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMSparsityFormat;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "sparsity_format";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: SparsityFormat,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMSparsityFormatAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) SparsityFormat {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMSparsityFormatAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.state_space`: NVVM State Space
pub const StateSpaceAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMStateSpace;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "state_space";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: StateSpace,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMStateSpaceAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) StateSpace {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMStateSpaceAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.store_cache_modifier`: NVVM store cache modifier kind
pub const StoreCacheModifierKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMStoreCacheModifierKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "store_cache_modifier";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: StoreCacheModifierKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMStoreCacheModifierKindAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) StoreCacheModifierKind {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMStoreCacheModifierKindAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.store_shape`: shape attribute for stmatrix
pub const StoreShapeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMStoreShape;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "store_shape";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: StoreShape,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMStoreShapeAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) StoreShape {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMStoreShapeAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.TCBarParam`: Cluster MMA Barrier Parameter Type
pub const TCBarParamAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTCBarParam;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "TCBarParam";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: TCBarParam,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTCBarParamAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) TCBarParam {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMTCBarParamAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.tma_load_mode`: NVVM TMA Load Mode
pub const TMALoadModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTMALoadMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "tma_load_mode";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: TMALoadMode,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTMALoadModeAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) TMALoadMode {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMTMALoadModeAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.tma_redux_kind`: NVVM TMA redux kind
pub const TMAReduxKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTMAReduxKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "tma_redux_kind";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: TMAReduxKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTMAReduxKindAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) TMAReduxKind {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMTMAReduxKindAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.tma_store_mode`: NVVM TMA Store Mode
pub const TMAStoreModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTMAStoreMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "tma_store_mode";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: TMAStoreMode,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTMAStoreModeAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) TMAStoreMode {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMTMAStoreModeAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.tcgen05_cp_multicast`: tcgen05 cp multicast
pub const Tcgen05CpMulticastAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTcgen05CpMulticast;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "tcgen05_cp_multicast";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: Tcgen05CpMulticast,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTcgen05CpMulticastAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) Tcgen05CpMulticast {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMTcgen05CpMulticastAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.tcgen05_cp_shape`: tcgen05 cp shapes
pub const Tcgen05CpShapeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTcgen05CpShape;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "tcgen05_cp_shape";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: Tcgen05CpShape,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTcgen05CpShapeAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) Tcgen05CpShape {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMTcgen05CpShapeAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.tcgen05_cp_src_fmt`: tcgen05 cp source format
pub const Tcgen05CpSrcFormatAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTcgen05CpSrcFormat;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "tcgen05_cp_src_fmt";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: Tcgen05CpSrcFormat,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTcgen05CpSrcFormatAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) Tcgen05CpSrcFormat {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMTcgen05CpSrcFormatAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.tcgen05_fence`: NVVM Tcgen05 fence kind
pub const Tcgen05FenceKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTcgen05FenceKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "tcgen05_fence";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: Tcgen05FenceKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTcgen05FenceKindAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) Tcgen05FenceKind {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMTcgen05FenceKindAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.tcgen05_ldst_shape`: allowed 32-bit signless integer cases: 0, 1, 2, 3, 4
pub const Tcgen05LdStShapeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTcgen05LdStShape;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "tcgen05_ldst_shape";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: Tcgen05LdStShape,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTcgen05LdStShapeAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) Tcgen05LdStShape {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMTcgen05LdStShapeAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.tcgen05_mma_block_scale`: tcgen05.mma block scale attribute
pub const Tcgen05MMABlockScaleAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTcgen05MMABlockScale;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "tcgen05_mma_block_scale";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: Tcgen05MMABlockScale,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTcgen05MMABlockScaleAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) Tcgen05MMABlockScale {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMTcgen05MMABlockScaleAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.tcgen05_mma_collectorb`: tcgen05 MMA Collector Buffer B Attribute
pub const Tcgen05MMACollectorBBufferAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTcgen05MMACollectorBBuffer;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "tcgen05_mma_collectorb";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: Tcgen05MMACollectorBBuffer,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTcgen05MMACollectorBBufferAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) Tcgen05MMACollectorBBuffer {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMTcgen05MMACollectorBBufferAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.tcgen05_mma_collectorop`: tcgen05.mma Collector Buffer Operation
pub const Tcgen05MMACollectorOpAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTcgen05MMACollectorOp;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "tcgen05_mma_collectorop";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: Tcgen05MMACollectorOp,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTcgen05MMACollectorOpAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) Tcgen05MMACollectorOp {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMTcgen05MMACollectorOpAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.tcgen05_mma_kind`: tcgen05 MMA Supported Types
pub const Tcgen05MMAKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTcgen05MMAKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "tcgen05_mma_kind";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: Tcgen05MMAKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTcgen05MMAKindAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) Tcgen05MMAKind {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMTcgen05MMAKindAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.tcgen05_wait`: NVVM Tcgen05 wait kind
pub const Tcgen05WaitKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTcgen05WaitKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "tcgen05_wait";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: Tcgen05WaitKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTcgen05WaitKindAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) Tcgen05WaitKind {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMTcgen05WaitKindAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.tensormap_elemtype`: NVVM Tensormap Elemtype
pub const TensormapElemtypeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTensormapElemtype;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "tensormap_elemtype";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: TensormapElemtype,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTensormapElemtypeAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) TensormapElemtype {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMTensormapElemtypeAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.tensormap_field`: NVVM Tensormap Field Kind
pub const TensormapFieldAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTensormapField;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "tensormap_field";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: TensormapField,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTensormapFieldAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) TensormapField {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMTensormapFieldAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.tensormap_fill_mode`: NVVM Tensormap Fill Mode
pub const TensormapFillModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTensormapFillMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "tensormap_fill_mode";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: TensormapFillMode,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTensormapFillModeAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) TensormapFillMode {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMTensormapFillModeAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.tensormap_interleave_layout`: NVVM Tensormap Interleave Layout
pub const TensormapInterleaveLayoutAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTensormapInterleaveLayout;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "tensormap_interleave_layout";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: TensormapInterleaveLayout,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTensormapInterleaveLayoutAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) TensormapInterleaveLayout {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMTensormapInterleaveLayoutAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.tensormap_swizzle_atomicity`: NVVM Tensormap Swizzle Atomicity
pub const TensormapSwizzleAtomicityAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTensormapSwizzleAtomicity;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "tensormap_swizzle_atomicity";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: TensormapSwizzleAtomicity,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTensormapSwizzleAtomicityAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) TensormapSwizzleAtomicity {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMTensormapSwizzleAtomicityAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.tensormap_swizzle_mode`: NVVM Tensormap Swizzle Mode
pub const TensormapSwizzleModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTensormapSwizzleMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "tensormap_swizzle_mode";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: TensormapSwizzleMode,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTensormapSwizzleModeAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) TensormapSwizzleMode {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMTensormapSwizzleModeAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.TmemLayout`: Tensor Memory Layout Enumerated Type
pub const TmemLayoutAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMTmemLayout;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "TmemLayout";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: TmemLayout,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMTmemLayoutAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) TmemLayout {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMTmemLayoutAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.validate_pattern`: NVVM validate data pattern
pub const ValidatePatternAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMValidatePattern;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "validate_pattern";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: ValidatePattern,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMValidatePatternAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) ValidatePattern {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMValidatePatternAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.vote_sync_kind`: NVVM vote sync kind
pub const VoteSyncKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMVoteSyncKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "vote_sync_kind";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: VoteSyncKind,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMVoteSyncKindAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) VoteSyncKind {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMVoteSyncKindAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.wgmma_scale_in`: WGMMA overflow options
pub const WGMMAScaleInAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMWGMMAScaleIn;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "wgmma_scale_in";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: WGMMAScaleIn,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMWGMMAScaleInAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) WGMMAScaleIn {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMWGMMAScaleInAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.wgmma_scale_out`: WGMMA input predicate
pub const WGMMAScaleOutAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMWGMMAScaleOut;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "wgmma_scale_out";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: WGMMAScaleOut,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMWGMMAScaleOutAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) WGMMAScaleOut {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMWGMMAScaleOutAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.wgmma_type`: NVVM WGMMA types
pub const WGMMATypesAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMWGMMATypes;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "wgmma_type";
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: WGMMATypes,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVVMWGMMATypesAttrGet(ctx.ptr(), @backingInt(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn value(self: *const Self) WGMMATypes {
        return @fromBackingInt(@intCast(c.mlirCuteNVVMWGMMATypesAttrGetValue(self.ptr())));
    }
};

/// `#nvvm.ld_st_matrix_shape`: Matrix shape of ldmatrix, stmatrix and movmatrix
pub const LdStMatrixShapeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVVMLdStMatrixShape;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "ld_st_matrix_shape";
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
    pub fn m(self: *const Self) c_int {
        return c.mlirCuteNVVMLdStMatrixShapeAttrGetM(self.ptr());
    }
    pub fn n(self: *const Self) c_int {
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
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "shape";
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
    pub fn m(self: *const Self) c_int {
        return c.mlirCuteNVVMMMAShapeAttrGetM(self.ptr());
    }
    pub fn n(self: *const Self) c_int {
        return c.mlirCuteNVVMMMAShapeAttrGetN(self.ptr());
    }
    pub fn k(self: *const Self) c_int {
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
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const mnemonic = "target";
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
    pub fn o(self: *const Self) c_int {
        return c.mlirCuteNVVMTargetAttrGetO(self.ptr());
    }
    pub fn triple(self: *const Self) []const u8 {
        const s = c.mlirCuteNVVMTargetAttrGetTriple(self.ptr());
        return if (s.length == 0) "" else mlir.string(s);
    }
    pub fn chip(self: *const Self) []const u8 {
        const s = c.mlirCuteNVVMTargetAttrGetChip(self.ptr());
        return if (s.length == 0) "" else mlir.string(s);
    }
    pub fn features(self: *const Self) []const u8 {
        const s = c.mlirCuteNVVMTargetAttrGetFeatures(self.ptr());
        return if (s.length == 0) "" else mlir.string(s);
    }
    pub fn flags(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVVMTargetAttrGetFlags(self.ptr()).ptr);
    }
    pub fn link(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVVMTargetAttrGetLink(self.ptr()).ptr);
    }
    pub fn verifyTarget(self: *const Self) bool {
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
pub fn add_packed_f32x2(
    ctx: *mlir.Context,
    srcA: *const mlir.Value,
    srcB: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        rnd: ?*const mlir.Attribute = null,
        ftz: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (opts.ftz) |value| attributes.appendAssumeCapacity(.named(ctx, "ftz", value));
    return mlir.Operation.make(ctx, "nvvm.add.packed.f32x2", .{
        .operands = .{ .flat = &.{ srcA, srcB } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.add.packed.f32x2.bf16x2.f32x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn add_packed_f32x2_bf16x2_f32x2(
    ctx: *mlir.Context,
    srcA: *const mlir.Value,
    srcB: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        rnd: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    return mlir.Operation.make(ctx, "nvvm.add.packed.f32x2.bf16x2.f32x2", .{
        .operands = .{ .flat = &.{ srcA, srcB } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.add.packed.f32x2.f16x2.f32x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn add_packed_f32x2_f16x2_f32x2(
    ctx: *mlir.Context,
    srcA: *const mlir.Value,
    srcB: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        rnd: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    return mlir.Operation.make(ctx, "nvvm.add.packed.f32x2.f16x2.f32x2", .{
        .operands = .{ .flat = &.{ srcA, srcB } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.addf`. Result types are explicit; attributes use mlir.Attribute.
pub fn addf(
    ctx: *mlir.Context,
    lhs: *const mlir.Value,
    rhs: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        rnd: ?*const mlir.Attribute = null,
        sat: ?*const mlir.Attribute = null,
        ftz: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (opts.sat) |value| attributes.appendAssumeCapacity(.named(ctx, "sat", value));
    if (opts.ftz) |value| attributes.appendAssumeCapacity(.named(ctx, "ftz", value));
    return mlir.Operation.make(ctx, "nvvm.addf", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.applypriority.async.bulk`. Result types are explicit; attributes use mlir.Attribute.
pub fn applypriority_async_bulk(
    ctx: *mlir.Context,
    ptr: *const mlir.Value,
    size: *const mlir.Value,
    opts: struct {
        evict: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.evict) |value| attributes.appendAssumeCapacity(.named(ctx, "evict", value));
    return mlir.Operation.make(ctx, "nvvm.applypriority.async.bulk", .{
        .operands = .{ .flat = &.{ ptr, size } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.applypriority.async.bulk.tensor`. Result types are explicit; attributes use mlir.Attribute.
pub fn applypriority_async_bulk_tensor(
    ctx: *mlir.Context,
    tmaDesc: *const mlir.Value,
    coordinates: []const *const mlir.Value,
    im2colArgs: []const *const mlir.Value,
    opts: struct {
        mode: ?*const mlir.Attribute = null,
        evict: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.mode) |value| attributes.appendAssumeCapacity(.named(ctx, "mode", value));
    if (opts.evict) |value| attributes.appendAssumeCapacity(.named(ctx, "evict", value));
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
pub fn applypriority_async_bulk_tensor_override(
    ctx: *mlir.Context,
    tmaDesc: *const mlir.Value,
    overrideAdrr: *const mlir.Value,
    coordinates: []const *const mlir.Value,
    tensorSize: []const *const mlir.Value,
    lowerStride: []const *const mlir.Value,
    im2colArgs: []const *const mlir.Value,
    opts: struct {
        upperStride: ?*const mlir.Value = null,
        mode: ?*const mlir.Attribute = null,
        evict: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.mode) |value| attributes.appendAssumeCapacity(.named(ctx, "mode", value));
    if (opts.evict) |value| attributes.appendAssumeCapacity(.named(ctx, "evict", value));
    return mlir.Operation.make(ctx, "nvvm.applypriority.async.bulk.tensor.override", .{
        .operands = .{ .variadic = &.{
            &.{tmaDesc},
            &.{overrideAdrr},
            coordinates,
            tensorSize,
            lowerStride,
            if (opts.upperStride) |value| &.{value} else &.{},
            im2colArgs,
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.atomicrmw`. Result types are explicit; attributes use mlir.Attribute.
pub fn atomicrmw(
    ctx: *mlir.Context,
    ptr: *const mlir.Value,
    a: *const mlir.Value,
    res_type: *const mlir.Type,
    op: *const mlir.Attribute,
    opts: struct {
        b: ?*const mlir.Value = null,
        memOrder: ?*const mlir.Attribute = null,
        syncscope: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "op", op));
    if (opts.memOrder) |value| attributes.appendAssumeCapacity(.named(ctx, "memOrder", value));
    if (opts.syncscope) |value| attributes.appendAssumeCapacity(.named(ctx, "syncscope", value));
    return mlir.Operation.make(ctx, "nvvm.atomicrmw", .{
        .operands = .{ .flat = if (opts.b) |value| &.{ ptr, a, value } else &.{ ptr, a } },
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
pub fn barrier(
    ctx: *mlir.Context,
    opts: struct {
        barrierId: ?*const mlir.Value = null,
        numberOfThreads: ?*const mlir.Value = null,
        reductionPredicate: ?*const mlir.Value = null,
        res_type: ?*const mlir.Type = null,
        reductionOp: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var results: stdx.BoundedArray(*const mlir.Type, 256) = .empty;
    if (opts.res_type) |value| results.appendAssumeCapacity(value);
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.reductionOp) |value| attributes.appendAssumeCapacity(.named(ctx, "reductionOp", value));
    return mlir.Operation.make(ctx, "nvvm.barrier", .{
        .operands = .{ .variadic = &.{
            if (opts.barrierId) |value| &.{value} else &.{},
            if (opts.numberOfThreads) |value| &.{value} else &.{},
            if (opts.reductionPredicate) |value| &.{value} else &.{},
        } },
        .results = .{ .flat = results.constSlice() },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.barrier.arrive`. Result types are explicit; attributes use mlir.Attribute.
pub fn barrier_arrive(
    ctx: *mlir.Context,
    numberOfThreads: *const mlir.Value,
    opts: struct {
        barrierId: ?*const mlir.Value = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.barrier.arrive", .{
        .operands = .{ .flat = if (opts.barrierId) |value| &.{ value, numberOfThreads } else &.{numberOfThreads} },
        .location = location,
    });
}

/// `nvvm.barrier.cta.arrive`. Result types are explicit; attributes use mlir.Attribute.
pub fn barrier_cta_arrive(
    ctx: *mlir.Context,
    barrierId: *const mlir.Value,
    threadCount: *const mlir.Value,
    opts: struct {
        aligned: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.aligned) |value| attributes.appendAssumeCapacity(.named(ctx, "aligned", value));
    return mlir.Operation.make(ctx, "nvvm.barrier.cta.arrive", .{
        .operands = .{ .flat = &.{ barrierId, threadCount } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.barrier.cta.red`. Result types are explicit; attributes use mlir.Attribute.
pub fn barrier_cta_red(
    ctx: *mlir.Context,
    pred: *const mlir.Value,
    barrierId: *const mlir.Value,
    res_type: *const mlir.Type,
    kind: *const mlir.Attribute,
    opts: struct {
        threadCount: ?*const mlir.Value = null,
        aligned: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (opts.aligned) |value| attributes.appendAssumeCapacity(.named(ctx, "aligned", value));
    return mlir.Operation.make(ctx, "nvvm.barrier.cta.red", .{
        .operands = .{ .flat = if (opts.threadCount) |value| &.{ pred, barrierId, value } else &.{ pred, barrierId } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.barrier.cta.sync`. Result types are explicit; attributes use mlir.Attribute.
pub fn barrier_cta_sync(
    ctx: *mlir.Context,
    barrierId: *const mlir.Value,
    opts: struct {
        threadCount: ?*const mlir.Value = null,
        aligned: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.aligned) |value| attributes.appendAssumeCapacity(.named(ctx, "aligned", value));
    return mlir.Operation.make(ctx, "nvvm.barrier.cta.sync", .{
        .operands = .{ .flat = if (opts.threadCount) |value| &.{ barrierId, value } else &.{barrierId} },
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
pub fn clmad(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c_: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        high: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.high) |value| attributes.appendAssumeCapacity(.named(ctx, "high", value));
    return mlir.Operation.make(ctx, "nvvm.clmad", .{
        .operands = .{ .flat = &.{ a, b, c_ } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cluster.arrive`. Result types are explicit; attributes use mlir.Attribute.
pub fn cluster_arrive(
    ctx: *mlir.Context,
    opts: struct {
        aligned: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.aligned) |value| attributes.appendAssumeCapacity(.named(ctx, "aligned", value));
    return mlir.Operation.make(ctx, "nvvm.cluster.arrive", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cluster.arrive.relaxed`. Result types are explicit; attributes use mlir.Attribute.
pub fn cluster_arrive_relaxed(
    ctx: *mlir.Context,
    opts: struct {
        aligned: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.aligned) |value| attributes.appendAssumeCapacity(.named(ctx, "aligned", value));
    return mlir.Operation.make(ctx, "nvvm.cluster.arrive.relaxed", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cluster.wait`. Result types are explicit; attributes use mlir.Attribute.
pub fn cluster_wait(
    ctx: *mlir.Context,
    opts: struct {
        aligned: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.aligned) |value| attributes.appendAssumeCapacity(.named(ctx, "aligned", value));
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
pub fn clusterlaunchcontrol_try_cancel(
    ctx: *mlir.Context,
    smemAddress: *const mlir.Value,
    mbarrier: *const mlir.Value,
    opts: struct {
        multicast: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.multicast) |value| attributes.appendAssumeCapacity(.named(ctx, "multicast", value));
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
pub fn convert_and_pack_integer(
    ctx: *mlir.Context,
    srcA: *const mlir.Value,
    srcB: *const mlir.Value,
    dst_type: *const mlir.Type,
    convert_type: *const mlir.Attribute,
    opts: struct {
        srcC: ?*const mlir.Value = null,
        is_signed: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.is_signed) |value| attributes.appendAssumeCapacity(.named(ctx, "is_signed", value));
    attributes.appendAssumeCapacity(.named(ctx, "convert_type", convert_type));
    return mlir.Operation.make(ctx, "nvvm.convert.and.pack.integer", .{
        .operands = .{ .flat = if (opts.srcC) |value| &.{ srcA, srcB, value } else &.{ srcA, srcB } },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.bf16x2.to.f4x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_bf16x2_to_f4x2(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    dst_type: *const mlir.Type,
    dstTy: *const mlir.Attribute,
    opts: struct {
        scale_factor: ?*const mlir.Value = null,
        isPZO: ?*const mlir.Attribute = null,
        scale_factor_kind: ?*const mlir.Attribute = null,
        rnd: ?*const mlir.Attribute = null,
        relu: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    if (opts.isPZO) |value| attributes.appendAssumeCapacity(.named(ctx, "isPZO", value));
    if (opts.scale_factor_kind) |value| attributes.appendAssumeCapacity(.named(ctx, "scale_factor_kind", value));
    if (opts.rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (opts.relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    attributes.appendAssumeCapacity(.named(ctx, "dstTy", dstTy));
    return mlir.Operation.make(ctx, "nvvm.convert.bf16x2.to.f4x2", .{
        .operands = .{ .flat = if (opts.scale_factor) |value| &.{ src, value } else &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.bf16x2.to.f6x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_bf16x2_to_f6x2(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    dst_type: *const mlir.Type,
    dstTy: *const mlir.Attribute,
    opts: struct {
        scale_factor: ?*const mlir.Value = null,
        isPZO: ?*const mlir.Attribute = null,
        scale_factor_kind: ?*const mlir.Attribute = null,
        rnd: ?*const mlir.Attribute = null,
        relu: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    if (opts.isPZO) |value| attributes.appendAssumeCapacity(.named(ctx, "isPZO", value));
    if (opts.scale_factor_kind) |value| attributes.appendAssumeCapacity(.named(ctx, "scale_factor_kind", value));
    if (opts.rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (opts.relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    attributes.appendAssumeCapacity(.named(ctx, "dstTy", dstTy));
    return mlir.Operation.make(ctx, "nvvm.convert.bf16x2.to.f6x2", .{
        .operands = .{ .flat = if (opts.scale_factor) |value| &.{ src, value } else &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.bf16x2.to.f8x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_bf16x2_to_f8x2(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    dst_type: *const mlir.Type,
    dstTy: *const mlir.Attribute,
    opts: struct {
        scale_factor: ?*const mlir.Value = null,
        isPZO: ?*const mlir.Attribute = null,
        scale_factor_kind: ?*const mlir.Attribute = null,
        rnd: ?*const mlir.Attribute = null,
        sat: ?*const mlir.Attribute = null,
        relu: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 6) = .empty;
    if (opts.isPZO) |value| attributes.appendAssumeCapacity(.named(ctx, "isPZO", value));
    if (opts.scale_factor_kind) |value| attributes.appendAssumeCapacity(.named(ctx, "scale_factor_kind", value));
    if (opts.rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (opts.sat) |value| attributes.appendAssumeCapacity(.named(ctx, "sat", value));
    if (opts.relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    attributes.appendAssumeCapacity(.named(ctx, "dstTy", dstTy));
    return mlir.Operation.make(ctx, "nvvm.convert.bf16x2.to.f8x2", .{
        .operands = .{ .flat = if (opts.scale_factor) |value| &.{ src, value } else &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.bf16x2.to.s2f6x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_bf16x2_to_s2f6x2(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    dst_type: *const mlir.Type,
    opts: struct {
        scaleFactor: ?*const mlir.Value = null,
        relu: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    return mlir.Operation.make(ctx, "nvvm.convert.bf16x2.to.s2f6x2", .{
        .operands = .{ .flat = if (opts.scaleFactor) |value| &.{ src, value } else &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.f16x2.to.f4x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_f16x2_to_f4x2(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    dst_type: *const mlir.Type,
    dstTy: *const mlir.Attribute,
    opts: struct {
        scale_factor: ?*const mlir.Value = null,
        isPZO: ?*const mlir.Attribute = null,
        scale_factor_kind: ?*const mlir.Attribute = null,
        rnd: ?*const mlir.Attribute = null,
        relu: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    if (opts.isPZO) |value| attributes.appendAssumeCapacity(.named(ctx, "isPZO", value));
    if (opts.scale_factor_kind) |value| attributes.appendAssumeCapacity(.named(ctx, "scale_factor_kind", value));
    if (opts.rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (opts.relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    attributes.appendAssumeCapacity(.named(ctx, "dstTy", dstTy));
    return mlir.Operation.make(ctx, "nvvm.convert.f16x2.to.f4x2", .{
        .operands = .{ .flat = if (opts.scale_factor) |value| &.{ src, value } else &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.f16x2.to.f6x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_f16x2_to_f6x2(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    dst_type: *const mlir.Type,
    dstTy: *const mlir.Attribute,
    opts: struct {
        scale_factor: ?*const mlir.Value = null,
        isPZO: ?*const mlir.Attribute = null,
        scale_factor_kind: ?*const mlir.Attribute = null,
        rnd: ?*const mlir.Attribute = null,
        relu: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    if (opts.isPZO) |value| attributes.appendAssumeCapacity(.named(ctx, "isPZO", value));
    if (opts.scale_factor_kind) |value| attributes.appendAssumeCapacity(.named(ctx, "scale_factor_kind", value));
    if (opts.rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (opts.relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    attributes.appendAssumeCapacity(.named(ctx, "dstTy", dstTy));
    return mlir.Operation.make(ctx, "nvvm.convert.f16x2.to.f6x2", .{
        .operands = .{ .flat = if (opts.scale_factor) |value| &.{ src, value } else &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.f16x2.to.f8x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_f16x2_to_f8x2(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    dst_type: *const mlir.Type,
    dstTy: *const mlir.Attribute,
    opts: struct {
        scale_factor: ?*const mlir.Value = null,
        isPZO: ?*const mlir.Attribute = null,
        scale_factor_kind: ?*const mlir.Attribute = null,
        rnd: ?*const mlir.Attribute = null,
        sat: ?*const mlir.Attribute = null,
        relu: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 6) = .empty;
    if (opts.isPZO) |value| attributes.appendAssumeCapacity(.named(ctx, "isPZO", value));
    if (opts.scale_factor_kind) |value| attributes.appendAssumeCapacity(.named(ctx, "scale_factor_kind", value));
    if (opts.rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (opts.sat) |value| attributes.appendAssumeCapacity(.named(ctx, "sat", value));
    if (opts.relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    attributes.appendAssumeCapacity(.named(ctx, "dstTy", dstTy));
    return mlir.Operation.make(ctx, "nvvm.convert.f16x2.to.f8x2", .{
        .operands = .{ .flat = if (opts.scale_factor) |value| &.{ a, value } else &.{a} },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.f32x2.to.bf16x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_f32x2_to_bf16x2(
    ctx: *mlir.Context,
    src_hi: *const mlir.Value,
    src_lo: *const mlir.Value,
    dst_type: *const mlir.Type,
    opts: struct {
        random_bits: ?*const mlir.Value = null,
        rnd: ?*const mlir.Attribute = null,
        sat: ?*const mlir.Attribute = null,
        relu: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (opts.sat) |value| attributes.appendAssumeCapacity(.named(ctx, "sat", value));
    if (opts.relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    return mlir.Operation.make(ctx, "nvvm.convert.f32x2.to.bf16x2", .{
        .operands = .{ .flat = if (opts.random_bits) |value| &.{ src_hi, src_lo, value } else &.{ src_hi, src_lo } },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.f32x2.to.f16x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_f32x2_to_f16x2(
    ctx: *mlir.Context,
    src_hi: *const mlir.Value,
    src_lo: *const mlir.Value,
    dst_type: *const mlir.Type,
    opts: struct {
        random_bits: ?*const mlir.Value = null,
        rnd: ?*const mlir.Attribute = null,
        sat: ?*const mlir.Attribute = null,
        relu: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (opts.sat) |value| attributes.appendAssumeCapacity(.named(ctx, "sat", value));
    if (opts.relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    return mlir.Operation.make(ctx, "nvvm.convert.f32x2.to.f16x2", .{
        .operands = .{ .flat = if (opts.random_bits) |value| &.{ src_hi, src_lo, value } else &.{ src_hi, src_lo } },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.f32x2.to.f4x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_f32x2_to_f4x2(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    dst_type: *const mlir.Type,
    dstType: *const mlir.Attribute,
    opts: struct {
        scale_factor: ?*const mlir.Value = null,
        isPZO: ?*const mlir.Attribute = null,
        scale_factor_kind: ?*const mlir.Attribute = null,
        rnd: ?*const mlir.Attribute = null,
        relu: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    if (opts.isPZO) |value| attributes.appendAssumeCapacity(.named(ctx, "isPZO", value));
    if (opts.scale_factor_kind) |value| attributes.appendAssumeCapacity(.named(ctx, "scale_factor_kind", value));
    if (opts.rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (opts.relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    attributes.appendAssumeCapacity(.named(ctx, "dstType", dstType));
    return mlir.Operation.make(ctx, "nvvm.convert.f32x2.to.f4x2", .{
        .operands = .{ .flat = if (opts.scale_factor) |value| &.{ a, b, value } else &.{ a, b } },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.f32x2.to.f6x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_f32x2_to_f6x2(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    dst_type: *const mlir.Type,
    dstTy: *const mlir.Attribute,
    opts: struct {
        scale_factor: ?*const mlir.Value = null,
        isPZO: ?*const mlir.Attribute = null,
        scale_factor_kind: ?*const mlir.Attribute = null,
        rnd: ?*const mlir.Attribute = null,
        relu: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    if (opts.isPZO) |value| attributes.appendAssumeCapacity(.named(ctx, "isPZO", value));
    if (opts.scale_factor_kind) |value| attributes.appendAssumeCapacity(.named(ctx, "scale_factor_kind", value));
    if (opts.rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (opts.relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    attributes.appendAssumeCapacity(.named(ctx, "dstTy", dstTy));
    return mlir.Operation.make(ctx, "nvvm.convert.f32x2.to.f6x2", .{
        .operands = .{ .flat = if (opts.scale_factor) |value| &.{ a, b, value } else &.{ a, b } },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.f32x2.to.f8x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_f32x2_to_f8x2(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    dst_type: *const mlir.Type,
    dstTy: *const mlir.Attribute,
    opts: struct {
        scale_factor: ?*const mlir.Value = null,
        isPZO: ?*const mlir.Attribute = null,
        scale_factor_kind: ?*const mlir.Attribute = null,
        rnd: ?*const mlir.Attribute = null,
        sat: ?*const mlir.Attribute = null,
        relu: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 6) = .empty;
    if (opts.isPZO) |value| attributes.appendAssumeCapacity(.named(ctx, "isPZO", value));
    if (opts.scale_factor_kind) |value| attributes.appendAssumeCapacity(.named(ctx, "scale_factor_kind", value));
    if (opts.rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (opts.sat) |value| attributes.appendAssumeCapacity(.named(ctx, "sat", value));
    if (opts.relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    attributes.appendAssumeCapacity(.named(ctx, "dstTy", dstTy));
    return mlir.Operation.make(ctx, "nvvm.convert.f32x2.to.f8x2", .{
        .operands = .{ .flat = if (opts.scale_factor) |value| &.{ a, b, value } else &.{ a, b } },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.f32x2.to.s2f6x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_f32x2_to_s2f6x2(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    dst_type: *const mlir.Type,
    opts: struct {
        scaleFactor: ?*const mlir.Value = null,
        relu: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    return mlir.Operation.make(ctx, "nvvm.convert.f32x2.to.s2f6x2", .{
        .operands = .{ .flat = if (opts.scaleFactor) |value| &.{ a, b, value } else &.{ a, b } },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.f32x4.to.f4x4`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_f32x4_to_f4x4(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    rbits: *const mlir.Value,
    dst_type: *const mlir.Type,
    dstTy: *const mlir.Attribute,
    opts: struct {
        relu: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    attributes.appendAssumeCapacity(.named(ctx, "dstTy", dstTy));
    return mlir.Operation.make(ctx, "nvvm.convert.f32x4.to.f4x4", .{
        .operands = .{ .flat = &.{ src, rbits } },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.f32x4.to.f6x4`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_f32x4_to_f6x4(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    rbits: *const mlir.Value,
    dst_type: *const mlir.Type,
    dstTy: *const mlir.Attribute,
    opts: struct {
        relu: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    attributes.appendAssumeCapacity(.named(ctx, "dstTy", dstTy));
    return mlir.Operation.make(ctx, "nvvm.convert.f32x4.to.f6x4", .{
        .operands = .{ .flat = &.{ src, rbits } },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.f32x4.to.f8x4`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_f32x4_to_f8x4(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    rbits: *const mlir.Value,
    dst_type: *const mlir.Type,
    dstTy: *const mlir.Attribute,
    opts: struct {
        relu: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    attributes.appendAssumeCapacity(.named(ctx, "dstTy", dstTy));
    return mlir.Operation.make(ctx, "nvvm.convert.f32x4.to.f8x4", .{
        .operands = .{ .flat = &.{ src, rbits } },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.f4x2.to.f16x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_f4x2_to_f16x2(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    dst_type: *const mlir.Type,
    srcType: *const mlir.Attribute,
    opts: struct {
        relu: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    attributes.appendAssumeCapacity(.named(ctx, "srcType", srcType));
    return mlir.Operation.make(ctx, "nvvm.convert.f4x2.to.f16x2", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.f6x2.to.f16x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_f6x2_to_f16x2(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    dst_type: *const mlir.Type,
    srcType: *const mlir.Attribute,
    opts: struct {
        relu: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    attributes.appendAssumeCapacity(.named(ctx, "srcType", srcType));
    return mlir.Operation.make(ctx, "nvvm.convert.f6x2.to.f16x2", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.f8x2.to.bf16x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_f8x2_to_bf16x2(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    dst_type: *const mlir.Type,
    srcType: *const mlir.Attribute,
    opts: struct {
        scaleFactor: ?*const mlir.Value = null,
        sat: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.sat) |value| attributes.appendAssumeCapacity(.named(ctx, "sat", value));
    attributes.appendAssumeCapacity(.named(ctx, "srcType", srcType));
    return mlir.Operation.make(ctx, "nvvm.convert.f8x2.to.bf16x2", .{
        .operands = .{ .flat = if (opts.scaleFactor) |value| &.{ src, value } else &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.f8x2.to.f16x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_f8x2_to_f16x2(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    dst_type: *const mlir.Type,
    srcType: *const mlir.Attribute,
    opts: struct {
        relu: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    attributes.appendAssumeCapacity(.named(ctx, "srcType", srcType));
    return mlir.Operation.make(ctx, "nvvm.convert.f8x2.to.f16x2", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.float.to.integer`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_float_to_integer(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    dst_type: *const mlir.Type,
    opts: struct {
        rnd: ?*const mlir.Attribute = null,
        sat: ?*const mlir.Attribute = null,
        ftz: ?*const mlir.Attribute = null,
        is_signed: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    if (opts.rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (opts.sat) |value| attributes.appendAssumeCapacity(.named(ctx, "sat", value));
    if (opts.ftz) |value| attributes.appendAssumeCapacity(.named(ctx, "ftz", value));
    if (opts.is_signed) |value| attributes.appendAssumeCapacity(.named(ctx, "is_signed", value));
    return mlir.Operation.make(ctx, "nvvm.convert.float.to.integer", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.float.to.tf32`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_float_to_tf32(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        rnd: ?*const mlir.Attribute = null,
        sat: ?*const mlir.Attribute = null,
        relu: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (opts.sat) |value| attributes.appendAssumeCapacity(.named(ctx, "sat", value));
    if (opts.relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    return mlir.Operation.make(ctx, "nvvm.convert.float.to.tf32", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.convert.s2f6x2.to.bf16x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn convert_s2f6x2_to_bf16x2(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    dst_type: *const mlir.Type,
    opts: struct {
        scaleFactor: ?*const mlir.Value = null,
        sat: ?*const mlir.Attribute = null,
        relu: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.sat) |value| attributes.appendAssumeCapacity(.named(ctx, "sat", value));
    if (opts.relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    return mlir.Operation.make(ctx, "nvvm.convert.s2f6x2.to.bf16x2", .{
        .operands = .{ .flat = if (opts.scaleFactor) |value| &.{ src, value } else &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cos`. Result types are explicit; attributes use mlir.Attribute.
pub fn cos(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        ftz: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.ftz) |value| attributes.appendAssumeCapacity(.named(ctx, "ftz", value));
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
pub fn cp_async_bulk_global_shared_cta(
    ctx: *mlir.Context,
    dstMem: *const mlir.Value,
    srcMem: *const mlir.Value,
    size: *const mlir.Value,
    opts: struct {
        l2CacheHint: ?*const mlir.Value = null,
        byteMask: ?*const mlir.Value = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.cp.async.bulk.global.shared.cta", .{
        .operands = .{ .variadic = &.{
            &.{dstMem},
            &.{srcMem},
            &.{size},
            if (opts.l2CacheHint) |value| &.{value} else &.{},
            if (opts.byteMask) |value| &.{value} else &.{},
        } },
        .location = location,
    });
}

/// `nvvm.cp.async.bulk.prefetch`. Result types are explicit; attributes use mlir.Attribute.
pub fn cp_async_bulk_prefetch(
    ctx: *mlir.Context,
    srcMem: *const mlir.Value,
    size: *const mlir.Value,
    opts: struct {
        l2CacheHint: ?*const mlir.Value = null,
        evict: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.evict) |value| attributes.appendAssumeCapacity(.named(ctx, "evict", value));
    return mlir.Operation.make(ctx, "nvvm.cp.async.bulk.prefetch", .{
        .operands = .{ .flat = if (opts.l2CacheHint) |value| &.{ srcMem, size, value } else &.{ srcMem, size } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cp.async.bulk.shared.cluster.global`. Result types are explicit; attributes use mlir.Attribute.
pub fn cp_async_bulk_shared_cluster_global(
    ctx: *mlir.Context,
    dstMem: *const mlir.Value,
    srcMem: *const mlir.Value,
    mbar: *const mlir.Value,
    size: *const mlir.Value,
    opts: struct {
        multicastMask: ?*const mlir.Value = null,
        l2CacheHint: ?*const mlir.Value = null,
        validatePattern: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.validatePattern) |value| attributes.appendAssumeCapacity(.named(ctx, "validatePattern", value));
    return mlir.Operation.make(ctx, "nvvm.cp.async.bulk.shared.cluster.global", .{
        .operands = .{ .variadic = &.{
            &.{dstMem},
            &.{srcMem},
            &.{mbar},
            &.{size},
            if (opts.multicastMask) |value| &.{value} else &.{},
            if (opts.l2CacheHint) |value| &.{value} else &.{},
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
pub fn cp_async_bulk_tensor_global_shared_cta(
    ctx: *mlir.Context,
    tmaDescriptor: *const mlir.Value,
    srcMem: *const mlir.Value,
    coordinates: []const *const mlir.Value,
    opts: struct {
        l2CacheHint: ?*const mlir.Value = null,
        predicate: ?*const mlir.Value = null,
        mode: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.mode) |value| attributes.appendAssumeCapacity(.named(ctx, "mode", value));
    return mlir.Operation.make(ctx, "nvvm.cp.async.bulk.tensor.global.shared.cta", .{
        .operands = .{ .variadic = &.{
            &.{tmaDescriptor},
            &.{srcMem},
            coordinates,
            if (opts.l2CacheHint) |value| &.{value} else &.{},
            if (opts.predicate) |value| &.{value} else &.{},
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cp.async.bulk.tensor.global.shared.cta.override`. Result types are explicit; attributes use mlir.Attribute.
pub fn cp_async_bulk_tensor_global_shared_cta_override(
    ctx: *mlir.Context,
    tmaDesc: *const mlir.Value,
    srcMem: *const mlir.Value,
    overrideAdrr: *const mlir.Value,
    coordinates: []const *const mlir.Value,
    tensorSize: []const *const mlir.Value,
    lowerStride: []const *const mlir.Value,
    opts: struct {
        upperStride: ?*const mlir.Value = null,
        l2CacheHint: ?*const mlir.Value = null,
        mode: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.mode) |value| attributes.appendAssumeCapacity(.named(ctx, "mode", value));
    return mlir.Operation.make(ctx, "nvvm.cp.async.bulk.tensor.global.shared.cta.override", .{
        .operands = .{ .variadic = &.{
            &.{tmaDesc},
            &.{srcMem},
            &.{overrideAdrr},
            coordinates,
            tensorSize,
            lowerStride,
            if (opts.upperStride) |value| &.{value} else &.{},
            if (opts.l2CacheHint) |value| &.{value} else &.{},
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cp.async.bulk.tensor.prefetch`. Result types are explicit; attributes use mlir.Attribute.
pub fn cp_async_bulk_tensor_prefetch(
    ctx: *mlir.Context,
    tmaDescriptor: *const mlir.Value,
    coordinates: []const *const mlir.Value,
    im2colOffsets: []const *const mlir.Value,
    opts: struct {
        l2CacheHint: ?*const mlir.Value = null,
        mode: ?*const mlir.Attribute = null,
        evict: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.mode) |value| attributes.appendAssumeCapacity(.named(ctx, "mode", value));
    if (opts.evict) |value| attributes.appendAssumeCapacity(.named(ctx, "evict", value));
    return mlir.Operation.make(ctx, "nvvm.cp.async.bulk.tensor.prefetch", .{
        .operands = .{ .variadic = &.{
            &.{tmaDescriptor},
            coordinates,
            im2colOffsets,
            if (opts.l2CacheHint) |value| &.{value} else &.{},
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cp.async.bulk.tensor.prefetch.override`. Result types are explicit; attributes use mlir.Attribute.
pub fn cp_async_bulk_tensor_prefetch_override(
    ctx: *mlir.Context,
    tmaDesc: *const mlir.Value,
    overrideAdrr: *const mlir.Value,
    coordinates: []const *const mlir.Value,
    tensorSize: []const *const mlir.Value,
    lowerStride: []const *const mlir.Value,
    im2colOffsets: []const *const mlir.Value,
    opts: struct {
        upperStride: ?*const mlir.Value = null,
        l2CacheHint: ?*const mlir.Value = null,
        evict: ?*const mlir.Attribute = null,
        mode: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.evict) |value| attributes.appendAssumeCapacity(.named(ctx, "evict", value));
    if (opts.mode) |value| attributes.appendAssumeCapacity(.named(ctx, "mode", value));
    return mlir.Operation.make(ctx, "nvvm.cp.async.bulk.tensor.prefetch.override", .{
        .operands = .{ .variadic = &.{
            &.{tmaDesc},
            &.{overrideAdrr},
            coordinates,
            tensorSize,
            lowerStride,
            if (opts.upperStride) |value| &.{value} else &.{},
            im2colOffsets,
            if (opts.l2CacheHint) |value| &.{value} else &.{},
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cp.async.bulk.tensor.reduce`. Result types are explicit; attributes use mlir.Attribute.
pub fn cp_async_bulk_tensor_reduce(
    ctx: *mlir.Context,
    tmaDescriptor: *const mlir.Value,
    srcMem: *const mlir.Value,
    coordinates: []const *const mlir.Value,
    redKind: *const mlir.Attribute,
    opts: struct {
        l2CacheHint: ?*const mlir.Value = null,
        mode: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "redKind", redKind));
    if (opts.mode) |value| attributes.appendAssumeCapacity(.named(ctx, "mode", value));
    return mlir.Operation.make(ctx, "nvvm.cp.async.bulk.tensor.reduce", .{
        .operands = .{ .variadic = &.{
            &.{tmaDescriptor},
            &.{srcMem},
            coordinates,
            if (opts.l2CacheHint) |value| &.{value} else &.{},
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cp.async.bulk.tensor.reduce.override`. Result types are explicit; attributes use mlir.Attribute.
pub fn cp_async_bulk_tensor_reduce_override(
    ctx: *mlir.Context,
    tmaDesc: *const mlir.Value,
    srcMem: *const mlir.Value,
    overrideAdrr: *const mlir.Value,
    coordinates: []const *const mlir.Value,
    tensorSize: []const *const mlir.Value,
    lowerStride: []const *const mlir.Value,
    redKind: *const mlir.Attribute,
    opts: struct {
        upperStride: ?*const mlir.Value = null,
        l2CacheHint: ?*const mlir.Value = null,
        mode: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "redKind", redKind));
    if (opts.mode) |value| attributes.appendAssumeCapacity(.named(ctx, "mode", value));
    return mlir.Operation.make(ctx, "nvvm.cp.async.bulk.tensor.reduce.override", .{
        .operands = .{ .variadic = &.{
            &.{tmaDesc},
            &.{srcMem},
            &.{overrideAdrr},
            coordinates,
            tensorSize,
            lowerStride,
            if (opts.upperStride) |value| &.{value} else &.{},
            if (opts.l2CacheHint) |value| &.{value} else &.{},
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cp.async.bulk.tensor.shared.cluster.global`. Result types are explicit; attributes use mlir.Attribute.
pub fn cp_async_bulk_tensor_shared_cluster_global(
    ctx: *mlir.Context,
    dstMem: *const mlir.Value,
    tmaDescriptor: *const mlir.Value,
    coordinates: []const *const mlir.Value,
    mbar: *const mlir.Value,
    im2colOffsets: []const *const mlir.Value,
    opts: struct {
        multicastMask: ?*const mlir.Value = null,
        l2CacheHint: ?*const mlir.Value = null,
        predicate: ?*const mlir.Value = null,
        validatePattern: ?*const mlir.Attribute = null,
        mode: ?*const mlir.Attribute = null,
        isCTAOnly: ?*const mlir.Attribute = null,
        group: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    if (opts.validatePattern) |value| attributes.appendAssumeCapacity(.named(ctx, "validatePattern", value));
    if (opts.mode) |value| attributes.appendAssumeCapacity(.named(ctx, "mode", value));
    if (opts.isCTAOnly) |value| attributes.appendAssumeCapacity(.named(ctx, "isCTAOnly", value));
    if (opts.group) |value| attributes.appendAssumeCapacity(.named(ctx, "group", value));
    return mlir.Operation.make(ctx, "nvvm.cp.async.bulk.tensor.shared.cluster.global", .{
        .operands = .{ .variadic = &.{
            &.{dstMem},
            &.{tmaDescriptor},
            coordinates,
            &.{mbar},
            im2colOffsets,
            if (opts.multicastMask) |value| &.{value} else &.{},
            if (opts.l2CacheHint) |value| &.{value} else &.{},
            if (opts.predicate) |value| &.{value} else &.{},
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cp.async.bulk.tensor.shared.cluster.global.override`. Result types are explicit; attributes use mlir.Attribute.
pub fn cp_async_bulk_tensor_shared_cluster_global_override(
    ctx: *mlir.Context,
    dstMem: *const mlir.Value,
    tmaDesc: *const mlir.Value,
    mbar: *const mlir.Value,
    overrideAddr: *const mlir.Value,
    coordinates: []const *const mlir.Value,
    im2colArgs: []const *const mlir.Value,
    tensorSize: []const *const mlir.Value,
    lowerStride: []const *const mlir.Value,
    opts: struct {
        upperStride: ?*const mlir.Value = null,
        multicastMask: ?*const mlir.Value = null,
        l2CacheHint: ?*const mlir.Value = null,
        group: ?*const mlir.Attribute = null,
        validatePattern: ?*const mlir.Attribute = null,
        mode: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.group) |value| attributes.appendAssumeCapacity(.named(ctx, "group", value));
    if (opts.validatePattern) |value| attributes.appendAssumeCapacity(.named(ctx, "validatePattern", value));
    if (opts.mode) |value| attributes.appendAssumeCapacity(.named(ctx, "mode", value));
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
            if (opts.upperStride) |value| &.{value} else &.{},
            if (opts.multicastMask) |value| &.{value} else &.{},
            if (opts.l2CacheHint) |value| &.{value} else &.{},
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cp.async.bulk.wait_group`. Result types are explicit; attributes use mlir.Attribute.
pub fn cp_async_bulk_wait_group(
    ctx: *mlir.Context,
    group: *const mlir.Attribute,
    opts: struct {
        read: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "group", group));
    if (opts.read) |value| attributes.appendAssumeCapacity(.named(ctx, "read", value));
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
pub fn cp_async_mbarrier_arrive(
    ctx: *mlir.Context,
    addr: *const mlir.Value,
    opts: struct {
        noinc: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.noinc) |value| attributes.appendAssumeCapacity(.named(ctx, "noinc", value));
    return mlir.Operation.make(ctx, "nvvm.cp.async.mbarrier.arrive", .{
        .operands = .{ .flat = &.{addr} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cp.async.shared.global`. Result types are explicit; attributes use mlir.Attribute.
pub fn cp_async_shared_global(
    ctx: *mlir.Context,
    dst: *const mlir.Value,
    src: *const mlir.Value,
    size: *const mlir.Attribute,
    modifier: *const mlir.Attribute,
    opts: struct {
        cpSize: ?*const mlir.Value = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.cp.async.shared.global", .{
        .operands = .{ .flat = if (opts.cpSize) |value| &.{ dst, src, value } else &.{ dst, src } },
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
pub fn cvt_packfloat(
    ctx: *mlir.Context,
    srcA: *const mlir.Value,
    srcC: *const mlir.Value,
    res_type: *const mlir.Type,
    from: *const mlir.Attribute,
    to: *const mlir.Attribute,
    opts: struct {
        rnd: ?*const mlir.Attribute = null,
        sat: ?*const mlir.Attribute = null,
        relu: ?*const mlir.Attribute = null,
        extractHi: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 6) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "from", from));
    attributes.appendAssumeCapacity(.named(ctx, "to", to));
    if (opts.rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (opts.sat) |value| attributes.appendAssumeCapacity(.named(ctx, "sat", value));
    if (opts.relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    if (opts.extractHi) |value| attributes.appendAssumeCapacity(.named(ctx, "extractHi", value));
    return mlir.Operation.make(ctx, "nvvm.cvt.packfloat", .{
        .operands = .{ .flat = &.{ srcA, srcC } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cvt.packfloat.f32`. Result types are explicit; attributes use mlir.Attribute.
pub fn cvt_packfloat_f32(
    ctx: *mlir.Context,
    srcA: *const mlir.Value,
    srcB: *const mlir.Value,
    srcC: *const mlir.Value,
    res_type: *const mlir.Type,
    to: *const mlir.Attribute,
    opts: struct {
        rnd: ?*const mlir.Attribute = null,
        sat: ?*const mlir.Attribute = null,
        relu: ?*const mlir.Attribute = null,
        extractHi: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "to", to));
    if (opts.rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (opts.sat) |value| attributes.appendAssumeCapacity(.named(ctx, "sat", value));
    if (opts.relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    if (opts.extractHi) |value| attributes.appendAssumeCapacity(.named(ctx, "extractHi", value));
    return mlir.Operation.make(ctx, "nvvm.cvt.packfloat.f32", .{
        .operands = .{ .flat = &.{ srcA, srcB, srcC } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.cvt.to.f4x8.packed`. Result types are explicit; attributes use mlir.Attribute.
pub fn cvt_to_f4x8_packed(
    ctx: *mlir.Context,
    srcs: []const *const mlir.Value,
    dst_type: *const mlir.Type,
    srcType: *const mlir.Attribute,
    dstType: *const mlir.Attribute,
    opts: struct {
        rnd: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
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
pub fn elect_sync(
    ctx: *mlir.Context,
    pred_type: *const mlir.Type,
    opts: struct {
        membermask: ?*const mlir.Value = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.elect.sync", .{
        .operands = .{ .flat = if (opts.membermask) |value| &.{value} else &.{} },
        .results = .{ .flat = &.{pred_type} },
        .location = location,
    });
}

/// `nvvm.ex2`. Result types are explicit; attributes use mlir.Attribute.
pub fn ex2(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        ftz: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.ftz) |value| attributes.appendAssumeCapacity(.named(ctx, "ftz", value));
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
pub fn fabs(
    ctx: *mlir.Context,
    value_: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        ftz: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.ftz) |value| attributes.appendAssumeCapacity(.named(ctx, "ftz", value));
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
pub fn fence_proxy(
    ctx: *mlir.Context,
    kind: *const mlir.Attribute,
    opts: struct {
        space: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (opts.space) |value| attributes.appendAssumeCapacity(.named(ctx, "space", value));
    return mlir.Operation.make(ctx, "nvvm.fence.proxy", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.fence.proxy.acquire`. Result types are explicit; attributes use mlir.Attribute.
pub fn fence_proxy_acquire(
    ctx: *mlir.Context,
    addr: *const mlir.Value,
    size: *const mlir.Value,
    scope: *const mlir.Attribute,
    opts: struct {
        fromProxy: ?*const mlir.Attribute = null,
        toProxy: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "scope", scope));
    if (opts.fromProxy) |value| attributes.appendAssumeCapacity(.named(ctx, "fromProxy", value));
    if (opts.toProxy) |value| attributes.appendAssumeCapacity(.named(ctx, "toProxy", value));
    return mlir.Operation.make(ctx, "nvvm.fence.proxy.acquire", .{
        .operands = .{ .flat = &.{ addr, size } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.fence.proxy.release`. Result types are explicit; attributes use mlir.Attribute.
pub fn fence_proxy_release(
    ctx: *mlir.Context,
    scope: *const mlir.Attribute,
    opts: struct {
        fromProxy: ?*const mlir.Attribute = null,
        toProxy: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "scope", scope));
    if (opts.fromProxy) |value| attributes.appendAssumeCapacity(.named(ctx, "fromProxy", value));
    if (opts.toProxy) |value| attributes.appendAssumeCapacity(.named(ctx, "toProxy", value));
    return mlir.Operation.make(ctx, "nvvm.fence.proxy.release", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.fence.proxy.sync_restrict`. Result types are explicit; attributes use mlir.Attribute.
pub fn fence_proxy_sync_restrict(
    ctx: *mlir.Context,
    order: *const mlir.Attribute,
    opts: struct {
        fromProxy: ?*const mlir.Attribute = null,
        toProxy: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "order", order));
    if (opts.fromProxy) |value| attributes.appendAssumeCapacity(.named(ctx, "fromProxy", value));
    if (opts.toProxy) |value| attributes.appendAssumeCapacity(.named(ctx, "toProxy", value));
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
pub fn fma(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c_: *const mlir.Value,
    res_type: *const mlir.Type,
    rnd: *const mlir.Attribute,
    opts: struct {
        sat: ?*const mlir.Attribute = null,
        ftz: ?*const mlir.Attribute = null,
        relu: ?*const mlir.Attribute = null,
        oob: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "rnd", rnd));
    if (opts.sat) |value| attributes.appendAssumeCapacity(.named(ctx, "sat", value));
    if (opts.ftz) |value| attributes.appendAssumeCapacity(.named(ctx, "ftz", value));
    if (opts.relu) |value| attributes.appendAssumeCapacity(.named(ctx, "relu", value));
    if (opts.oob) |value| attributes.appendAssumeCapacity(.named(ctx, "oob", value));
    return mlir.Operation.make(ctx, "nvvm.fma", .{
        .operands = .{ .flat = &.{ a, b, c_ } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.fma.packed.f32x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn fma_packed_f32x2(
    ctx: *mlir.Context,
    srcA: *const mlir.Value,
    srcB: *const mlir.Value,
    srcC: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        rnd: ?*const mlir.Attribute = null,
        ftz: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (opts.ftz) |value| attributes.appendAssumeCapacity(.named(ctx, "ftz", value));
    return mlir.Operation.make(ctx, "nvvm.fma.packed.f32x2", .{
        .operands = .{ .flat = &.{ srcA, srcB, srcC } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.fma.packed.f32x2.bf16x2.f32x2.f32x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn fma_packed_f32x2_bf16x2_f32x2_f32x2(
    ctx: *mlir.Context,
    srcA: *const mlir.Value,
    srcB: *const mlir.Value,
    srcC: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        rnd: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    return mlir.Operation.make(ctx, "nvvm.fma.packed.f32x2.bf16x2.f32x2.f32x2", .{
        .operands = .{ .flat = &.{ srcA, srcB, srcC } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.fma.packed.f32x2.f16x2.f32x2.f32x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn fma_packed_f32x2_f16x2_f32x2_f32x2(
    ctx: *mlir.Context,
    srcA: *const mlir.Value,
    srcB: *const mlir.Value,
    srcC: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        rnd: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    return mlir.Operation.make(ctx, "nvvm.fma.packed.f32x2.f16x2.f32x2.f32x2", .{
        .operands = .{ .flat = &.{ srcA, srcB, srcC } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.fmax`. Result types are explicit; attributes use mlir.Attribute.
pub fn fmax(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        c_: ?*const mlir.Value = null,
        ftz: ?*const mlir.Attribute = null,
        nan: ?*const mlir.Attribute = null,
        abs: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.ftz) |value| attributes.appendAssumeCapacity(.named(ctx, "ftz", value));
    if (opts.nan) |value| attributes.appendAssumeCapacity(.named(ctx, "nan", value));
    if (opts.abs) |value| attributes.appendAssumeCapacity(.named(ctx, "abs", value));
    return mlir.Operation.make(ctx, "nvvm.fmax", .{
        .operands = .{ .flat = if (opts.c_) |value| &.{ a, b, value } else &.{ a, b } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.fmin`. Result types are explicit; attributes use mlir.Attribute.
pub fn fmin(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        c_: ?*const mlir.Value = null,
        ftz: ?*const mlir.Attribute = null,
        nan: ?*const mlir.Attribute = null,
        abs: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.ftz) |value| attributes.appendAssumeCapacity(.named(ctx, "ftz", value));
    if (opts.nan) |value| attributes.appendAssumeCapacity(.named(ctx, "nan", value));
    if (opts.abs) |value| attributes.appendAssumeCapacity(.named(ctx, "abs", value));
    return mlir.Operation.make(ctx, "nvvm.fmin", .{
        .operands = .{ .flat = if (opts.c_) |value| &.{ a, b, value } else &.{ a, b } },
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
pub fn inline_ptx(
    ctx: *mlir.Context,
    readOnlyArgs: []const *const mlir.Value,
    readWriteArgs: []const *const mlir.Value,
    ptxCode: *const mlir.Attribute,
    opts: struct {
        predicate: ?*const mlir.Value = null,
        writeOnlyArgs_type: ?*const mlir.Type = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var results: stdx.BoundedArray(*const mlir.Type, 256) = .empty;
    if (opts.writeOnlyArgs_type) |value| results.appendAssumeCapacity(value);
    return mlir.Operation.make(ctx, "nvvm.inline_ptx", .{
        .operands = .{ .variadic = &.{
            readOnlyArgs,
            readWriteArgs,
            if (opts.predicate) |value| &.{value} else &.{},
        } },
        .results = .{ .flat = results.constSlice() },
        .attributes = &.{
            .named(ctx, "ptxCode", ptxCode),
        },
        .location = location,
    });
}

/// `nvvm.ldmatrix`. Result types are explicit; attributes use mlir.Attribute.
pub fn ldmatrix(
    ctx: *mlir.Context,
    ptr: *const mlir.Value,
    res_type: *const mlir.Type,
    num: *const mlir.Attribute,
    layout: *const mlir.Attribute,
    opts: struct {
        shape: ?*const mlir.Attribute = null,
        srcFormat: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "num", num));
    attributes.appendAssumeCapacity(.named(ctx, "layout", layout));
    if (opts.shape) |value| attributes.appendAssumeCapacity(.named(ctx, "shape", value));
    if (opts.srcFormat) |value| attributes.appendAssumeCapacity(.named(ctx, "srcFormat", value));
    return mlir.Operation.make(ctx, "nvvm.ldmatrix", .{
        .operands = .{ .flat = &.{ptr} },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.load.ext`. Result types are explicit; attributes use mlir.Attribute.
pub fn load_ext(
    ctx: *mlir.Context,
    addr: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        l2CacheHint: ?*const mlir.Value = null,
        order: ?*const mlir.Attribute = null,
        scope: ?*const mlir.Attribute = null,
        prefetch_: ?*const mlir.Attribute = null,
        evict: ?*const mlir.Attribute = null,
        cacheModifier: ?*const mlir.Attribute = null,
        sharedSpace: ?*const mlir.Attribute = null,
        unified: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 7) = .empty;
    if (opts.order) |value| attributes.appendAssumeCapacity(.named(ctx, "order", value));
    if (opts.scope) |value| attributes.appendAssumeCapacity(.named(ctx, "scope", value));
    if (opts.prefetch_) |value| attributes.appendAssumeCapacity(.named(ctx, "prefetch", value));
    if (opts.evict) |value| attributes.appendAssumeCapacity(.named(ctx, "evict", value));
    if (opts.cacheModifier) |value| attributes.appendAssumeCapacity(.named(ctx, "cacheModifier", value));
    if (opts.sharedSpace) |value| attributes.appendAssumeCapacity(.named(ctx, "sharedSpace", value));
    if (opts.unified) |value| attributes.appendAssumeCapacity(.named(ctx, "unified", value));
    return mlir.Operation.make(ctx, "nvvm.load.ext", .{
        .operands = .{ .flat = if (opts.l2CacheHint) |value| &.{ addr, value } else &.{addr} },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.log2`. Result types are explicit; attributes use mlir.Attribute.
pub fn log2(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        ftz: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.ftz) |value| attributes.appendAssumeCapacity(.named(ctx, "ftz", value));
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
pub fn mbarrier_arrive(
    ctx: *mlir.Context,
    addr: *const mlir.Value,
    opts: struct {
        count: ?*const mlir.Value = null,
        multicastMask: ?*const mlir.Value = null,
        res_type: ?*const mlir.Type = null,
        scope: ?*const mlir.Attribute = null,
        relaxed: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var results: stdx.BoundedArray(*const mlir.Type, 256) = .empty;
    if (opts.res_type) |value| results.appendAssumeCapacity(value);
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.scope) |value| attributes.appendAssumeCapacity(.named(ctx, "scope", value));
    if (opts.relaxed) |value| attributes.appendAssumeCapacity(.named(ctx, "relaxed", value));
    return mlir.Operation.make(ctx, "nvvm.mbarrier.arrive", .{
        .operands = .{ .variadic = &.{
            &.{addr},
            if (opts.count) |value| &.{value} else &.{},
            if (opts.multicastMask) |value| &.{value} else &.{},
        } },
        .results = .{ .flat = results.constSlice() },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mbarrier.arrive.expect_tx`. Result types are explicit; attributes use mlir.Attribute.
pub fn mbarrier_arrive_expect_tx(
    ctx: *mlir.Context,
    addr: *const mlir.Value,
    txcount: *const mlir.Value,
    opts: struct {
        multicastMask: ?*const mlir.Value = null,
        predicate: ?*const mlir.Value = null,
        res_type: ?*const mlir.Type = null,
        scope: ?*const mlir.Attribute = null,
        relaxed: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var results: stdx.BoundedArray(*const mlir.Type, 256) = .empty;
    if (opts.res_type) |value| results.appendAssumeCapacity(value);
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.scope) |value| attributes.appendAssumeCapacity(.named(ctx, "scope", value));
    if (opts.relaxed) |value| attributes.appendAssumeCapacity(.named(ctx, "relaxed", value));
    return mlir.Operation.make(ctx, "nvvm.mbarrier.arrive.expect_tx", .{
        .operands = .{ .variadic = &.{
            &.{addr},
            &.{txcount},
            if (opts.multicastMask) |value| &.{value} else &.{},
            if (opts.predicate) |value| &.{value} else &.{},
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
pub fn mbarrier_arrive_drop(
    ctx: *mlir.Context,
    addr: *const mlir.Value,
    opts: struct {
        count: ?*const mlir.Value = null,
        multicastMask: ?*const mlir.Value = null,
        res_type: ?*const mlir.Type = null,
        scope: ?*const mlir.Attribute = null,
        relaxed: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var results: stdx.BoundedArray(*const mlir.Type, 256) = .empty;
    if (opts.res_type) |value| results.appendAssumeCapacity(value);
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.scope) |value| attributes.appendAssumeCapacity(.named(ctx, "scope", value));
    if (opts.relaxed) |value| attributes.appendAssumeCapacity(.named(ctx, "relaxed", value));
    return mlir.Operation.make(ctx, "nvvm.mbarrier.arrive_drop", .{
        .operands = .{ .variadic = &.{
            &.{addr},
            if (opts.count) |value| &.{value} else &.{},
            if (opts.multicastMask) |value| &.{value} else &.{},
        } },
        .results = .{ .flat = results.constSlice() },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mbarrier.arrive_drop.expect_tx`. Result types are explicit; attributes use mlir.Attribute.
pub fn mbarrier_arrive_drop_expect_tx(
    ctx: *mlir.Context,
    addr: *const mlir.Value,
    txcount: *const mlir.Value,
    opts: struct {
        multicastMask: ?*const mlir.Value = null,
        res_type: ?*const mlir.Type = null,
        scope: ?*const mlir.Attribute = null,
        relaxed: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var results: stdx.BoundedArray(*const mlir.Type, 256) = .empty;
    if (opts.res_type) |value| results.appendAssumeCapacity(value);
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.scope) |value| attributes.appendAssumeCapacity(.named(ctx, "scope", value));
    if (opts.relaxed) |value| attributes.appendAssumeCapacity(.named(ctx, "relaxed", value));
    return mlir.Operation.make(ctx, "nvvm.mbarrier.arrive_drop.expect_tx", .{
        .operands = .{ .flat = if (opts.multicastMask) |value| &.{ addr, txcount, value } else &.{ addr, txcount } },
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
pub fn mbarrier_complete_tx(
    ctx: *mlir.Context,
    addr: *const mlir.Value,
    txcount: *const mlir.Value,
    opts: struct {
        report: ?*const mlir.Value = null,
        multicastMask: ?*const mlir.Value = null,
        scope: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.scope) |value| attributes.appendAssumeCapacity(.named(ctx, "scope", value));
    return mlir.Operation.make(ctx, "nvvm.mbarrier.complete_tx", .{
        .operands = .{ .variadic = &.{
            &.{addr},
            &.{txcount},
            if (opts.report) |value| &.{value} else &.{},
            if (opts.multicastMask) |value| &.{value} else &.{},
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mbarrier.expect_tx`. Result types are explicit; attributes use mlir.Attribute.
pub fn mbarrier_expect_tx(
    ctx: *mlir.Context,
    addr: *const mlir.Value,
    txcount: *const mlir.Value,
    opts: struct {
        multicastMask: ?*const mlir.Value = null,
        scope: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.scope) |value| attributes.appendAssumeCapacity(.named(ctx, "scope", value));
    return mlir.Operation.make(ctx, "nvvm.mbarrier.expect_tx", .{
        .operands = .{ .flat = if (opts.multicastMask) |value| &.{ addr, txcount, value } else &.{ addr, txcount } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mbarrier.init`. Result types are explicit; attributes use mlir.Attribute.
pub fn mbarrier_init(
    ctx: *mlir.Context,
    addr: *const mlir.Value,
    count: *const mlir.Value,
    opts: struct {
        predicate: ?*const mlir.Value = null,
        layout: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.layout) |value| attributes.appendAssumeCapacity(.named(ctx, "layout", value));
    return mlir.Operation.make(ctx, "nvvm.mbarrier.init", .{
        .operands = .{ .flat = if (opts.predicate) |value| &.{ addr, count, value } else &.{ addr, count } },
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
pub fn mbarrier_test_wait(
    ctx: *mlir.Context,
    addr: *const mlir.Value,
    stateOrPhase: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        scope: ?*const mlir.Attribute = null,
        relaxed: ?*const mlir.Attribute = null,
        mbarrierPhase: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.scope) |value| attributes.appendAssumeCapacity(.named(ctx, "scope", value));
    if (opts.relaxed) |value| attributes.appendAssumeCapacity(.named(ctx, "relaxed", value));
    if (opts.mbarrierPhase) |value| attributes.appendAssumeCapacity(.named(ctx, "mbarrierPhase", value));
    return mlir.Operation.make(ctx, "nvvm.mbarrier.test.wait", .{
        .operands = .{ .flat = &.{ addr, stateOrPhase } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mbarrier.try_wait`. Result types are explicit; attributes use mlir.Attribute.
pub fn mbarrier_try_wait(
    ctx: *mlir.Context,
    addr: *const mlir.Value,
    stateOrPhase: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        ticks: ?*const mlir.Value = null,
        scope: ?*const mlir.Attribute = null,
        relaxed: ?*const mlir.Attribute = null,
        mbarrierPhase: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.scope) |value| attributes.appendAssumeCapacity(.named(ctx, "scope", value));
    if (opts.relaxed) |value| attributes.appendAssumeCapacity(.named(ctx, "relaxed", value));
    if (opts.mbarrierPhase) |value| attributes.appendAssumeCapacity(.named(ctx, "mbarrierPhase", value));
    return mlir.Operation.make(ctx, "nvvm.mbarrier.try_wait", .{
        .operands = .{ .flat = if (opts.ticks) |value| &.{ addr, stateOrPhase, value } else &.{ addr, stateOrPhase } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mbarrier.try_wait.parity`. Result types are explicit; attributes use mlir.Attribute.
pub fn mbarrier_try_wait_parity(
    ctx: *mlir.Context,
    addr: *const mlir.Value,
    phase: *const mlir.Value,
    ticks: *const mlir.Value,
    opts: struct {
        useIntrinsic: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.useIntrinsic) |value| attributes.appendAssumeCapacity(.named(ctx, "useIntrinsic", value));
    return mlir.Operation.make(ctx, "nvvm.mbarrier.try_wait.parity", .{
        .operands = .{ .flat = &.{ addr, phase, ticks } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mbarrier.try_wait.parity.timelimit`. Result types are explicit; attributes use mlir.Attribute.
pub fn mbarrier_try_wait_parity_timelimit(
    ctx: *mlir.Context,
    addr: *const mlir.Value,
    phase: *const mlir.Value,
    timeLimit: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        scope: ?*const mlir.Attribute = null,
        order: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.scope) |value| attributes.appendAssumeCapacity(.named(ctx, "scope", value));
    if (opts.order) |value| attributes.appendAssumeCapacity(.named(ctx, "order", value));
    return mlir.Operation.make(ctx, "nvvm.mbarrier.try_wait.parity.timelimit", .{
        .operands = .{ .flat = &.{ addr, phase, timeLimit } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mbarrier.try_wait.timelimit`. Result types are explicit; attributes use mlir.Attribute.
pub fn mbarrier_try_wait_timelimit(
    ctx: *mlir.Context,
    addr: *const mlir.Value,
    state: *const mlir.Value,
    timeLimit: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        scope: ?*const mlir.Attribute = null,
        order: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.scope) |value| attributes.appendAssumeCapacity(.named(ctx, "scope", value));
    if (opts.order) |value| attributes.appendAssumeCapacity(.named(ctx, "order", value));
    return mlir.Operation.make(ctx, "nvvm.mbarrier.try_wait.timelimit", .{
        .operands = .{ .flat = &.{ addr, state, timeLimit } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mbarrier.txn`. Result types are explicit; attributes use mlir.Attribute.
pub fn mbarrier_txn(
    ctx: *mlir.Context,
    addr: *const mlir.Value,
    count: *const mlir.Value,
    kind: *const mlir.Attribute,
    opts: struct {
        multicast: ?*const mlir.Value = null,
        space: ?*const mlir.Attribute = null,
        scope: ?*const mlir.Attribute = null,
        order: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (opts.space) |value| attributes.appendAssumeCapacity(.named(ctx, "space", value));
    if (opts.scope) |value| attributes.appendAssumeCapacity(.named(ctx, "scope", value));
    if (opts.order) |value| attributes.appendAssumeCapacity(.named(ctx, "order", value));
    return mlir.Operation.make(ctx, "nvvm.mbarrier.txn", .{
        .operands = .{ .flat = if (opts.multicast) |value| &.{ addr, count, value } else &.{ addr, count } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mbarrier.txn.cta`. Result types are explicit; attributes use mlir.Attribute.
pub fn mbarrier_txn_cta(
    ctx: *mlir.Context,
    addr: *const mlir.Value,
    count: *const mlir.Value,
    res_type: *const mlir.Type,
    kind: *const mlir.Attribute,
    opts: struct {
        scope: ?*const mlir.Attribute = null,
        order: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (opts.scope) |value| attributes.appendAssumeCapacity(.named(ctx, "scope", value));
    if (opts.order) |value| attributes.appendAssumeCapacity(.named(ctx, "order", value));
    return mlir.Operation.make(ctx, "nvvm.mbarrier.txn.cta", .{
        .operands = .{ .flat = &.{ addr, count } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mbarrier.wait`. Result types are explicit; attributes use mlir.Attribute.
pub fn mbarrier_wait(
    ctx: *mlir.Context,
    addr: *const mlir.Value,
    state: *const mlir.Value,
    res_type: *const mlir.Type,
    kind: *const mlir.Attribute,
    opts: struct {
        scope: ?*const mlir.Attribute = null,
        order: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (opts.scope) |value| attributes.appendAssumeCapacity(.named(ctx, "scope", value));
    if (opts.order) |value| attributes.appendAssumeCapacity(.named(ctx, "order", value));
    return mlir.Operation.make(ctx, "nvvm.mbarrier.wait", .{
        .operands = .{ .flat = &.{ addr, state } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mbarrier.wait.parity`. Result types are explicit; attributes use mlir.Attribute.
pub fn mbarrier_wait_parity(
    ctx: *mlir.Context,
    addr: *const mlir.Value,
    phase: *const mlir.Value,
    res_type: *const mlir.Type,
    kind: *const mlir.Attribute,
    opts: struct {
        scope: ?*const mlir.Attribute = null,
        order: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (opts.scope) |value| attributes.appendAssumeCapacity(.named(ctx, "scope", value));
    if (opts.order) |value| attributes.appendAssumeCapacity(.named(ctx, "order", value));
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
pub fn mma_block_scale(
    ctx: *mlir.Context,
    operandA: []const *const mlir.Value,
    operandB: []const *const mlir.Value,
    operandC: []const *const mlir.Value,
    scaleAData: *const mlir.Value,
    byteIdA: *const mlir.Value,
    threadIdA: *const mlir.Value,
    scaleBData: *const mlir.Value,
    byteIdB: *const mlir.Value,
    threadIdB: *const mlir.Value,
    res_type: *const mlir.Type,
    shape: *const mlir.Attribute,
    scaleVecSize: *const mlir.Attribute,
    blockScaleFormat: *const mlir.Attribute,
    kind: *const mlir.Attribute,
    opts: struct {
        multiplicandAPtxType: ?*const mlir.Attribute = null,
        multiplicandBPtxType: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 6) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "shape", shape));
    if (opts.multiplicandAPtxType) |value| attributes.appendAssumeCapacity(.named(ctx, "multiplicandAPtxType", value));
    if (opts.multiplicandBPtxType) |value| attributes.appendAssumeCapacity(.named(ctx, "multiplicandBPtxType", value));
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
pub fn mma_block_scale_internal(
    ctx: *mlir.Context,
    operandA: *const mlir.Value,
    scaleAData: *const mlir.Value,
    byteIdA: *const mlir.Value,
    threadIdA: *const mlir.Value,
    operandB: *const mlir.Value,
    scaleBData: *const mlir.Value,
    byteIdB: *const mlir.Value,
    threadIdB: *const mlir.Value,
    operandC: *const mlir.Value,
    res_type: *const mlir.Type,
    shape: *const mlir.Attribute,
    layoutA: *const mlir.Attribute,
    layoutB: *const mlir.Attribute,
    aType: *const mlir.Attribute,
    bType: *const mlir.Attribute,
    cType: *const mlir.Attribute,
    scaleVecSize: *const mlir.Attribute,
    blockScaleFormat: *const mlir.Attribute,
    opts: struct {
        kind: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 9) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "shape", shape));
    attributes.appendAssumeCapacity(.named(ctx, "layoutA", layoutA));
    attributes.appendAssumeCapacity(.named(ctx, "layoutB", layoutB));
    attributes.appendAssumeCapacity(.named(ctx, "aType", aType));
    attributes.appendAssumeCapacity(.named(ctx, "bType", bType));
    attributes.appendAssumeCapacity(.named(ctx, "cType", cType));
    attributes.appendAssumeCapacity(.named(ctx, "scaleVecSize", scaleVecSize));
    attributes.appendAssumeCapacity(.named(ctx, "blockScaleFormat", blockScaleFormat));
    if (opts.kind) |value| attributes.appendAssumeCapacity(.named(ctx, "kind", value));
    return mlir.Operation.make(ctx, "nvvm.mma.block_scale.internal", .{
        .operands = .{ .flat = &.{ operandA, scaleAData, byteIdA, threadIdA, operandB, scaleBData, byteIdB, threadIdB, operandC } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mma.sp.block_scale`. Result types are explicit; attributes use mlir.Attribute.
pub fn mma_sp_block_scale(
    ctx: *mlir.Context,
    operandA: []const *const mlir.Value,
    operandB: []const *const mlir.Value,
    operandC: []const *const mlir.Value,
    sparseMetadata: *const mlir.Value,
    sparsitySelector: *const mlir.Value,
    scaleAData: *const mlir.Value,
    byteIdA: *const mlir.Value,
    threadIdA: *const mlir.Value,
    scaleBData: *const mlir.Value,
    byteIdB: *const mlir.Value,
    threadIdB: *const mlir.Value,
    res_type: *const mlir.Type,
    shape: *const mlir.Attribute,
    scaleVecSize: *const mlir.Attribute,
    blockScaleFormat: *const mlir.Attribute,
    kind: *const mlir.Attribute,
    opts: struct {
        multiplicandAPtxType: ?*const mlir.Attribute = null,
        multiplicandBPtxType: ?*const mlir.Attribute = null,
        orderedMetadata: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 7) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "shape", shape));
    if (opts.multiplicandAPtxType) |value| attributes.appendAssumeCapacity(.named(ctx, "multiplicandAPtxType", value));
    if (opts.multiplicandBPtxType) |value| attributes.appendAssumeCapacity(.named(ctx, "multiplicandBPtxType", value));
    attributes.appendAssumeCapacity(.named(ctx, "scaleVecSize", scaleVecSize));
    attributes.appendAssumeCapacity(.named(ctx, "blockScaleFormat", blockScaleFormat));
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (opts.orderedMetadata) |value| attributes.appendAssumeCapacity(.named(ctx, "orderedMetadata", value));
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
pub fn mma_sp_block_scale_internal(
    ctx: *mlir.Context,
    operandA: *const mlir.Value,
    scaleAData: *const mlir.Value,
    byteIdA: *const mlir.Value,
    threadIdA: *const mlir.Value,
    operandB: *const mlir.Value,
    scaleBData: *const mlir.Value,
    byteIdB: *const mlir.Value,
    threadIdB: *const mlir.Value,
    operandC: *const mlir.Value,
    sp_metadata: *const mlir.Value,
    selector: *const mlir.Value,
    res_type: *const mlir.Type,
    shape: *const mlir.Attribute,
    layoutA: *const mlir.Attribute,
    layoutB: *const mlir.Attribute,
    aType: *const mlir.Attribute,
    bType: *const mlir.Attribute,
    cType: *const mlir.Attribute,
    scaleVecSize: *const mlir.Attribute,
    blockScaleFormat: *const mlir.Attribute,
    sparsityFormat: *const mlir.Attribute,
    opts: struct {
        b1Op: ?*const mlir.Attribute = null,
        intOverflowBehavior: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 11) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "shape", shape));
    if (opts.b1Op) |value| attributes.appendAssumeCapacity(.named(ctx, "b1Op", value));
    if (opts.intOverflowBehavior) |value| attributes.appendAssumeCapacity(.named(ctx, "intOverflowBehavior", value));
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
pub fn mma_sp_sync(
    ctx: *mlir.Context,
    operandA: []const *const mlir.Value,
    operandB: []const *const mlir.Value,
    operandC: []const *const mlir.Value,
    sparseMetadata: *const mlir.Value,
    sparsitySelector: *const mlir.Value,
    res_type: *const mlir.Type,
    shape: *const mlir.Attribute,
    opts: struct {
        intOverflowBehavior: ?*const mlir.Attribute = null,
        multiplicandAPtxType: ?*const mlir.Attribute = null,
        multiplicandBPtxType: ?*const mlir.Attribute = null,
        orderedMetadata: ?*const mlir.Attribute = null,
        kind: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 6) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "shape", shape));
    if (opts.intOverflowBehavior) |value| attributes.appendAssumeCapacity(.named(ctx, "intOverflowBehavior", value));
    if (opts.multiplicandAPtxType) |value| attributes.appendAssumeCapacity(.named(ctx, "multiplicandAPtxType", value));
    if (opts.multiplicandBPtxType) |value| attributes.appendAssumeCapacity(.named(ctx, "multiplicandBPtxType", value));
    if (opts.orderedMetadata) |value| attributes.appendAssumeCapacity(.named(ctx, "orderedMetadata", value));
    if (opts.kind) |value| attributes.appendAssumeCapacity(.named(ctx, "kind", value));
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
pub fn mma_sync(
    ctx: *mlir.Context,
    operandA: []const *const mlir.Value,
    operandB: []const *const mlir.Value,
    operandC: []const *const mlir.Value,
    res_type: *const mlir.Type,
    shape: *const mlir.Attribute,
    layoutA: *const mlir.Attribute,
    layoutB: *const mlir.Attribute,
    opts: struct {
        b1Op: ?*const mlir.Attribute = null,
        intOverflowBehavior: ?*const mlir.Attribute = null,
        multiplicandAPtxType: ?*const mlir.Attribute = null,
        multiplicandBPtxType: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 7) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "shape", shape));
    if (opts.b1Op) |value| attributes.appendAssumeCapacity(.named(ctx, "b1Op", value));
    if (opts.intOverflowBehavior) |value| attributes.appendAssumeCapacity(.named(ctx, "intOverflowBehavior", value));
    attributes.appendAssumeCapacity(.named(ctx, "layoutA", layoutA));
    attributes.appendAssumeCapacity(.named(ctx, "layoutB", layoutB));
    if (opts.multiplicandAPtxType) |value| attributes.appendAssumeCapacity(.named(ctx, "multiplicandAPtxType", value));
    if (opts.multiplicandBPtxType) |value| attributes.appendAssumeCapacity(.named(ctx, "multiplicandBPtxType", value));
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
pub fn mma_smem_desc(
    ctx: *mlir.Context,
    pointer: *const mlir.Value,
    ldm: *const mlir.Value,
    stride: *const mlir.Value,
    baseOffset: *const mlir.Value,
    swizzle: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        mmaDescVersion: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.mmaDescVersion) |value| attributes.appendAssumeCapacity(.named(ctx, "mmaDescVersion", value));
    return mlir.Operation.make(ctx, "nvvm.mma_smem_desc", .{
        .operands = .{ .flat = &.{ pointer, ldm, stride, baseOffset, swizzle } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.movmatrix`. Result types are explicit; attributes use mlir.Attribute.
pub fn movmatrix(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    dst_type: *const mlir.Type,
    shape: *const mlir.Attribute,
    eltType: *const mlir.Attribute,
    opts: struct {
        layout: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "shape", shape));
    if (opts.layout) |value| attributes.appendAssumeCapacity(.named(ctx, "layout", value));
    attributes.appendAssumeCapacity(.named(ctx, "eltType", eltType));
    return mlir.Operation.make(ctx, "nvvm.movmatrix", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.mul`. Result types are explicit; attributes use mlir.Attribute.
pub fn mul(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    res_type: *const mlir.Type,
    mode: *const mlir.Attribute,
    opts: struct {
        isSigned: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "mode", mode));
    if (opts.isSigned) |value| attributes.appendAssumeCapacity(.named(ctx, "isSigned", value));
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
pub fn mul_packed_f32x2(
    ctx: *mlir.Context,
    srcA: *const mlir.Value,
    srcB: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        rnd: ?*const mlir.Attribute = null,
        ftz: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (opts.ftz) |value| attributes.appendAssumeCapacity(.named(ctx, "ftz", value));
    return mlir.Operation.make(ctx, "nvvm.mul.packed.f32x2", .{
        .operands = .{ .flat = &.{ srcA, srcB } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.multimem.ld.reduce`. Result types are explicit; attributes use mlir.Attribute.
pub fn multimem_ld_reduce(
    ctx: *mlir.Context,
    addr: *const mlir.Value,
    res_type: *const mlir.Type,
    ordering: *const mlir.Attribute,
    op: *const mlir.Attribute,
    dataType: *const mlir.Attribute,
    opts: struct {
        accPrec: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "ordering", ordering));
    attributes.appendAssumeCapacity(.named(ctx, "op", op));
    attributes.appendAssumeCapacity(.named(ctx, "dataType", dataType));
    if (opts.accPrec) |value| attributes.appendAssumeCapacity(.named(ctx, "accPrec", value));
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
pub fn pmevent(
    ctx: *mlir.Context,
    opts: struct {
        maskedEventId: ?*const mlir.Attribute = null,
        eventId: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.maskedEventId) |value| attributes.appendAssumeCapacity(.named(ctx, "maskedEventId", value));
    if (opts.eventId) |value| attributes.appendAssumeCapacity(.named(ctx, "eventId", value));
    return mlir.Operation.make(ctx, "nvvm.pmevent", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.prefetch`. Result types are explicit; attributes use mlir.Attribute.
pub fn prefetch(
    ctx: *mlir.Context,
    addr: *const mlir.Value,
    opts: struct {
        predicate: ?*const mlir.Value = null,
        cacheLevel: ?*const mlir.Attribute = null,
        evictPriority: ?*const mlir.Attribute = null,
        tensormap: ?*const mlir.Attribute = null,
        uniform: ?*const mlir.Attribute = null,
        in_param_space: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    if (opts.cacheLevel) |value| attributes.appendAssumeCapacity(.named(ctx, "cacheLevel", value));
    if (opts.evictPriority) |value| attributes.appendAssumeCapacity(.named(ctx, "evictPriority", value));
    if (opts.tensormap) |value| attributes.appendAssumeCapacity(.named(ctx, "tensormap", value));
    if (opts.uniform) |value| attributes.appendAssumeCapacity(.named(ctx, "uniform", value));
    if (opts.in_param_space) |value| attributes.appendAssumeCapacity(.named(ctx, "in_param_space", value));
    return mlir.Operation.make(ctx, "nvvm.prefetch", .{
        .operands = .{ .flat = if (opts.predicate) |value| &.{ addr, value } else &.{addr} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.prmt`. Result types are explicit; attributes use mlir.Attribute.
pub fn prmt(
    ctx: *mlir.Context,
    lo: *const mlir.Value,
    selector: *const mlir.Value,
    res_type: *const mlir.Type,
    mode: *const mlir.Attribute,
    opts: struct {
        hi: ?*const mlir.Value = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.prmt", .{
        .operands = .{ .flat = if (opts.hi) |value| &.{ lo, value, selector } else &.{ lo, selector } },
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
pub fn read_ptx_sreg_cluster_ctaid_x(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.cluster.ctaid.x", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.cluster.ctaid.y`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_cluster_ctaid_y(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.cluster.ctaid.y", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.cluster.ctaid.z`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_cluster_ctaid_z(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.cluster.ctaid.z", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.cluster.ctarank`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_cluster_ctarank(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.cluster.ctarank", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.cluster.nctaid.x`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_cluster_nctaid_x(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.cluster.nctaid.x", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.cluster.nctaid.y`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_cluster_nctaid_y(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.cluster.nctaid.y", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.cluster.nctaid.z`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_cluster_nctaid_z(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.cluster.nctaid.z", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.cluster.nctarank`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_cluster_nctarank(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.cluster.nctarank", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.clusterid.x`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_clusterid_x(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.clusterid.x", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.clusterid.y`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_clusterid_y(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.clusterid.y", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.clusterid.z`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_clusterid_z(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.clusterid.z", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.ctaid.x`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_ctaid_x(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.ctaid.x", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.ctaid.y`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_ctaid_y(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.ctaid.y", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.ctaid.z`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_ctaid_z(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
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
pub fn read_ptx_sreg_gridid(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.gridid", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.laneid`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_laneid(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
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
pub fn read_ptx_sreg_nclusterid_x(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.nclusterid.x", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.nclusterid.y`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_nclusterid_y(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.nclusterid.y", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.nclusterid.z`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_nclusterid_z(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.nclusterid.z", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.nctaid.x`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_nctaid_x(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.nctaid.x", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.nctaid.y`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_nctaid_y(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.nctaid.y", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.nctaid.z`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_nctaid_z(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.nctaid.z", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.nsmid`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_nsmid(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.nsmid", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.ntid.x`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_ntid_x(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.ntid.x", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.ntid.y`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_ntid_y(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.ntid.y", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.ntid.z`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_ntid_z(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.ntid.z", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.nwarpid`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_nwarpid(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.nwarpid", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.smid`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_smid(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.smid", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.tid.x`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_tid_x(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.tid.x", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.tid.y`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_tid_y(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.tid.y", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.tid.z`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_tid_z(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
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
pub fn read_ptx_sreg_warpid(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.warpid", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.read.ptx.sreg.warpsize`. Result types are explicit; attributes use mlir.Attribute.
pub fn read_ptx_sreg_warpsize(
    ctx: *mlir.Context,
    res_type: *const mlir.Type,
    opts: struct {
        range: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.range) |value| attributes.appendAssumeCapacity(.named(ctx, "range", value));
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg.warpsize", .{
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.red`. Result types are explicit; attributes use mlir.Attribute.
pub fn red(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    op: *const mlir.Attribute,
    type_: *const mlir.Attribute,
    opts: struct {
        cacheHint: ?*const mlir.Value = null,
        memOrder: ?*const mlir.Attribute = null,
        memScope: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    if (opts.memOrder) |value| attributes.appendAssumeCapacity(.named(ctx, "memOrder", value));
    if (opts.memScope) |value| attributes.appendAssumeCapacity(.named(ctx, "memScope", value));
    attributes.appendAssumeCapacity(.named(ctx, "op", op));
    attributes.appendAssumeCapacity(.named(ctx, "type", type_));
    return mlir.Operation.make(ctx, "nvvm.red", .{
        .operands = .{ .flat = if (opts.cacheHint) |value| &.{ a, b, value } else &.{ a, b } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.redux.sync`. Result types are explicit; attributes use mlir.Attribute.
pub fn redux_sync(
    ctx: *mlir.Context,
    val: *const mlir.Value,
    mask_and_clamp: *const mlir.Value,
    res_type: *const mlir.Type,
    kind: *const mlir.Attribute,
    opts: struct {
        abs: ?*const mlir.Attribute = null,
        nan: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (opts.abs) |value| attributes.appendAssumeCapacity(.named(ctx, "abs", value));
    if (opts.nan) |value| attributes.appendAssumeCapacity(.named(ctx, "nan", value));
    return mlir.Operation.make(ctx, "nvvm.redux.sync", .{
        .operands = .{ .flat = &.{ val, mask_and_clamp } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.rsqrt`. Result types are explicit; attributes use mlir.Attribute.
pub fn rsqrt(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        ftz: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.ftz) |value| attributes.appendAssumeCapacity(.named(ctx, "ftz", value));
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
pub fn shfl_sync(
    ctx: *mlir.Context,
    thread_mask: *const mlir.Value,
    val: *const mlir.Value,
    offset: *const mlir.Value,
    mask_and_clamp: *const mlir.Value,
    res_type: *const mlir.Type,
    kind: *const mlir.Attribute,
    opts: struct {
        return_value_and_is_valid: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (opts.return_value_and_is_valid) |value| attributes.appendAssumeCapacity(.named(ctx, "return_value_and_is_valid", value));
    return mlir.Operation.make(ctx, "nvvm.shfl.sync", .{
        .operands = .{ .flat = &.{ thread_mask, val, offset, mask_and_clamp } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.sin`. Result types are explicit; attributes use mlir.Attribute.
pub fn sin(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        ftz: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.ftz) |value| attributes.appendAssumeCapacity(.named(ctx, "ftz", value));
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
pub fn st_bulk(
    ctx: *mlir.Context,
    addr: *const mlir.Value,
    size: *const mlir.Value,
    opts: struct {
        initVal: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.initVal) |value| attributes.appendAssumeCapacity(.named(ctx, "initVal", value));
    return mlir.Operation.make(ctx, "nvvm.st.bulk", .{
        .operands = .{ .flat = &.{ addr, size } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.stmatrix`. Result types are explicit; attributes use mlir.Attribute.
pub fn stmatrix(
    ctx: *mlir.Context,
    ptr: *const mlir.Value,
    sources: []const *const mlir.Value,
    layout: *const mlir.Attribute,
    opts: struct {
        shape: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 256) = .empty;
    operands.appendAssumeCapacity(ptr);
    operands.appendSliceAssumeCapacity(sources);
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "layout", layout));
    if (opts.shape) |value| attributes.appendAssumeCapacity(.named(ctx, "shape", value));
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
pub fn store_ext(
    ctx: *mlir.Context,
    value_: *const mlir.Value,
    addr: *const mlir.Value,
    opts: struct {
        l2CacheHint: ?*const mlir.Value = null,
        order: ?*const mlir.Attribute = null,
        scope: ?*const mlir.Attribute = null,
        evict: ?*const mlir.Attribute = null,
        cacheModifier: ?*const mlir.Attribute = null,
        sharedSpace: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    if (opts.order) |value| attributes.appendAssumeCapacity(.named(ctx, "order", value));
    if (opts.scope) |value| attributes.appendAssumeCapacity(.named(ctx, "scope", value));
    if (opts.evict) |value| attributes.appendAssumeCapacity(.named(ctx, "evict", value));
    if (opts.cacheModifier) |value| attributes.appendAssumeCapacity(.named(ctx, "cacheModifier", value));
    if (opts.sharedSpace) |value| attributes.appendAssumeCapacity(.named(ctx, "sharedSpace", value));
    return mlir.Operation.make(ctx, "nvvm.store.ext", .{
        .operands = .{ .flat = if (opts.l2CacheHint) |value| &.{ value_, addr, value } else &.{ value_, addr } },
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
pub fn sub_packed_f32x2(
    ctx: *mlir.Context,
    srcA: *const mlir.Value,
    srcB: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        rnd: ?*const mlir.Attribute = null,
        ftz: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (opts.ftz) |value| attributes.appendAssumeCapacity(.named(ctx, "ftz", value));
    return mlir.Operation.make(ctx, "nvvm.sub.packed.f32x2", .{
        .operands = .{ .flat = &.{ srcA, srcB } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.sub.packed.f32x2.bf16x2.f32x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn sub_packed_f32x2_bf16x2_f32x2(
    ctx: *mlir.Context,
    srcA: *const mlir.Value,
    srcB: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        rnd: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    return mlir.Operation.make(ctx, "nvvm.sub.packed.f32x2.bf16x2.f32x2", .{
        .operands = .{ .flat = &.{ srcA, srcB } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.sub.packed.f32x2.f16x2.f32x2`. Result types are explicit; attributes use mlir.Attribute.
pub fn sub_packed_f32x2_f16x2_f32x2(
    ctx: *mlir.Context,
    srcA: *const mlir.Value,
    srcB: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        rnd: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    return mlir.Operation.make(ctx, "nvvm.sub.packed.f32x2.f16x2.f32x2", .{
        .operands = .{ .flat = &.{ srcA, srcB } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.subf`. Result types are explicit; attributes use mlir.Attribute.
pub fn subf(
    ctx: *mlir.Context,
    lhs: *const mlir.Value,
    rhs: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: struct {
        rnd: ?*const mlir.Attribute = null,
        sat: ?*const mlir.Attribute = null,
        ftz: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (opts.rnd) |value| attributes.appendAssumeCapacity(.named(ctx, "rnd", value));
    if (opts.sat) |value| attributes.appendAssumeCapacity(.named(ctx, "sat", value));
    if (opts.ftz) |value| attributes.appendAssumeCapacity(.named(ctx, "ftz", value));
    return mlir.Operation.make(ctx, "nvvm.subf", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.alloc`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_alloc(
    ctx: *mlir.Context,
    addr: *const mlir.Value,
    nCols: *const mlir.Value,
    opts: struct {
        isExclusive: ?*const mlir.Attribute = null,
        group: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.isExclusive) |value| attributes.appendAssumeCapacity(.named(ctx, "isExclusive", value));
    if (opts.group) |value| attributes.appendAssumeCapacity(.named(ctx, "group", value));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.alloc", .{
        .operands = .{ .flat = &.{ addr, nCols } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.commit`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_commit(
    ctx: *mlir.Context,
    addr: *const mlir.Value,
    opts: struct {
        multicastMask: ?*const mlir.Value = null,
        smem_a_read: ?*const mlir.Attribute = null,
        group: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.smem_a_read) |value| attributes.appendAssumeCapacity(.named(ctx, "smem_a_read", value));
    if (opts.group) |value| attributes.appendAssumeCapacity(.named(ctx, "group", value));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.commit", .{
        .operands = .{ .flat = if (opts.multicastMask) |value| &.{ addr, value } else &.{addr} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.cp`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_cp(
    ctx: *mlir.Context,
    taddr: *const mlir.Value,
    smem_desc: *const mlir.Value,
    shape: *const mlir.Attribute,
    opts: struct {
        group: ?*const mlir.Attribute = null,
        multicast: ?*const mlir.Attribute = null,
        srcFormat: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "shape", shape));
    if (opts.group) |value| attributes.appendAssumeCapacity(.named(ctx, "group", value));
    if (opts.multicast) |value| attributes.appendAssumeCapacity(.named(ctx, "multicast", value));
    if (opts.srcFormat) |value| attributes.appendAssumeCapacity(.named(ctx, "srcFormat", value));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.cp", .{
        .operands = .{ .flat = &.{ taddr, smem_desc } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.dealloc`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_dealloc(
    ctx: *mlir.Context,
    taddr: *const mlir.Value,
    nCols: *const mlir.Value,
    opts: struct {
        isExclusive: ?*const mlir.Attribute = null,
        group: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.isExclusive) |value| attributes.appendAssumeCapacity(.named(ctx, "isExclusive", value));
    if (opts.group) |value| attributes.appendAssumeCapacity(.named(ctx, "group", value));
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
pub fn tcgen05_ld(
    ctx: *mlir.Context,
    tmemAddr: *const mlir.Value,
    res_type: *const mlir.Type,
    shape: *const mlir.Attribute,
    opts: struct {
        offset: ?*const mlir.Value = null,
        pack: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.pack) |value| attributes.appendAssumeCapacity(.named(ctx, "pack", value));
    attributes.appendAssumeCapacity(.named(ctx, "shape", shape));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.ld", .{
        .operands = .{ .flat = if (opts.offset) |value| &.{ tmemAddr, value } else &.{tmemAddr} },
        .results = .{ .flat = &.{res_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.ld.red`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_ld_red(
    ctx: *mlir.Context,
    addr: *const mlir.Value,
    data_type: *const mlir.Type,
    redVal_type: *const mlir.Type,
    shape: *const mlir.Attribute,
    op: *const mlir.Attribute,
    opts: struct {
        offset: ?*const mlir.Value = null,
        abs: ?*const mlir.Attribute = null,
        nan: ?*const mlir.Attribute = null,
        isSigned: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "shape", shape));
    attributes.appendAssumeCapacity(.named(ctx, "op", op));
    if (opts.abs) |value| attributes.appendAssumeCapacity(.named(ctx, "abs", value));
    if (opts.nan) |value| attributes.appendAssumeCapacity(.named(ctx, "nan", value));
    if (opts.isSigned) |value| attributes.appendAssumeCapacity(.named(ctx, "isSigned", value));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.ld.red", .{
        .operands = .{ .flat = if (opts.offset) |value| &.{ addr, value } else &.{addr} },
        .results = .{ .flat = &.{ data_type, redVal_type } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.ld.red.spcompress`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_ld_red_spcompress(
    ctx: *mlir.Context,
    tmemAddr: *const mlir.Value,
    metadata_type: *const mlir.Type,
    compressedData_type: *const mlir.Type,
    redVal_type: *const mlir.Type,
    shape: *const mlir.Attribute,
    opKind: *const mlir.Attribute,
    compFactor: *const mlir.Attribute,
    repFactor: *const mlir.Attribute,
    opts: struct {
        abs: ?*const mlir.Attribute = null,
        nan: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 6) = .empty;
    if (opts.abs) |value| attributes.appendAssumeCapacity(.named(ctx, "abs", value));
    if (opts.nan) |value| attributes.appendAssumeCapacity(.named(ctx, "nan", value));
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
pub fn tcgen05_ld_spcompress(
    ctx: *mlir.Context,
    tmemAddr: *const mlir.Value,
    metadata_type: *const mlir.Type,
    compressedData_type: *const mlir.Type,
    shape: *const mlir.Attribute,
    opKind: *const mlir.Attribute,
    compFactor: *const mlir.Attribute,
    repFactor: *const mlir.Attribute,
    opts: struct {
        abs: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    if (opts.abs) |value| attributes.appendAssumeCapacity(.named(ctx, "abs", value));
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
pub fn tcgen05_mma(
    ctx: *mlir.Context,
    matrixD: *const mlir.Value,
    matrixA: *const mlir.Value,
    matrixB: *const mlir.Value,
    idesc: *const mlir.Value,
    enableInputD: *const mlir.Value,
    kind: *const mlir.Attribute,
    ctaGroup: *const mlir.Attribute,
    opts: struct {
        scaleInputD: ?*const mlir.Value = null,
        disableOutputLane: ?*const mlir.Value = null,
        collectorOp: ?*const mlir.Attribute = null,
        collectorOpB: ?*const mlir.Attribute = null,
        aShift: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    attributes.appendAssumeCapacity(.named(ctx, "ctaGroup", ctaGroup));
    if (opts.collectorOp) |value| attributes.appendAssumeCapacity(.named(ctx, "collectorOp", value));
    if (opts.collectorOpB) |value| attributes.appendAssumeCapacity(.named(ctx, "collectorOpB", value));
    if (opts.aShift) |value| attributes.appendAssumeCapacity(.named(ctx, "aShift", value));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.mma", .{
        .operands = .{ .variadic = &.{
            &.{matrixD},
            &.{matrixA},
            &.{matrixB},
            &.{idesc},
            &.{enableInputD},
            if (opts.scaleInputD) |value| &.{value} else &.{},
            if (opts.disableOutputLane) |value| &.{value} else &.{},
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.mma.block_scale`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_mma_block_scale(
    ctx: *mlir.Context,
    matrixD: *const mlir.Value,
    matrixA: *const mlir.Value,
    matrixB: *const mlir.Value,
    idesc: *const mlir.Value,
    enableInputD: *const mlir.Value,
    scaleA: *const mlir.Value,
    scaleB: *const mlir.Value,
    kind: *const mlir.Attribute,
    ctaGroup: *const mlir.Attribute,
    opts: struct {
        blockScale: ?*const mlir.Attribute = null,
        collectorOp: ?*const mlir.Attribute = null,
        collectorOpB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    attributes.appendAssumeCapacity(.named(ctx, "ctaGroup", ctaGroup));
    if (opts.blockScale) |value| attributes.appendAssumeCapacity(.named(ctx, "blockScale", value));
    if (opts.collectorOp) |value| attributes.appendAssumeCapacity(.named(ctx, "collectorOp", value));
    if (opts.collectorOpB) |value| attributes.appendAssumeCapacity(.named(ctx, "collectorOpB", value));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.mma.block_scale", .{
        .operands = .{ .flat = &.{ matrixD, matrixA, matrixB, idesc, enableInputD, scaleA, scaleB } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.mma.block_scale.decompress_b`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_mma_block_scale_decompress_b(
    ctx: *mlir.Context,
    matrixD: *const mlir.Value,
    matrixA: *const mlir.Value,
    matrixB: *const mlir.Value,
    idesc: *const mlir.Value,
    enableInputD: *const mlir.Value,
    scaleA: *const mlir.Value,
    scaleB: *const mlir.Value,
    decompressBMetadata: *const mlir.Value,
    kind: *const mlir.Attribute,
    ctaGroup: *const mlir.Attribute,
    opts: struct {
        blockScale: ?*const mlir.Attribute = null,
        collectorOpA: ?*const mlir.Attribute = null,
        collectorOpB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    attributes.appendAssumeCapacity(.named(ctx, "ctaGroup", ctaGroup));
    if (opts.blockScale) |value| attributes.appendAssumeCapacity(.named(ctx, "blockScale", value));
    if (opts.collectorOpA) |value| attributes.appendAssumeCapacity(.named(ctx, "collectorOpA", value));
    if (opts.collectorOpB) |value| attributes.appendAssumeCapacity(.named(ctx, "collectorOpB", value));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.mma.block_scale.decompress_b", .{
        .operands = .{ .flat = &.{ matrixD, matrixA, matrixB, idesc, enableInputD, scaleA, scaleB, decompressBMetadata } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.mma.decompress_b`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_mma_decompress_b(
    ctx: *mlir.Context,
    matrixD: *const mlir.Value,
    matrixA: *const mlir.Value,
    matrixB: *const mlir.Value,
    idesc: *const mlir.Value,
    enableInputD: *const mlir.Value,
    decompressBMetadata: *const mlir.Value,
    kind: *const mlir.Attribute,
    ctaGroup: *const mlir.Attribute,
    opts: struct {
        disableOutputLane: ?*const mlir.Value = null,
        collectorOpA: ?*const mlir.Attribute = null,
        collectorOpB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    attributes.appendAssumeCapacity(.named(ctx, "ctaGroup", ctaGroup));
    if (opts.collectorOpA) |value| attributes.appendAssumeCapacity(.named(ctx, "collectorOpA", value));
    if (opts.collectorOpB) |value| attributes.appendAssumeCapacity(.named(ctx, "collectorOpB", value));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.mma.decompress_b", .{
        .operands = .{ .flat = if (opts.disableOutputLane) |value| &.{ matrixD, matrixA, matrixB, idesc, enableInputD, decompressBMetadata, value } else &.{ matrixD, matrixA, matrixB, idesc, enableInputD, decompressBMetadata } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.mma.sp`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_mma_sp(
    ctx: *mlir.Context,
    matrixD: *const mlir.Value,
    matrixA: *const mlir.Value,
    matrixB: *const mlir.Value,
    idesc: *const mlir.Value,
    enableInputD: *const mlir.Value,
    sparseMetadata: *const mlir.Value,
    kind: *const mlir.Attribute,
    ctaGroup: *const mlir.Attribute,
    opts: struct {
        scaleInputD: ?*const mlir.Value = null,
        disableOutputLane: ?*const mlir.Value = null,
        collectorOp: ?*const mlir.Attribute = null,
        collectorOpB: ?*const mlir.Attribute = null,
        aShift: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    attributes.appendAssumeCapacity(.named(ctx, "ctaGroup", ctaGroup));
    if (opts.collectorOp) |value| attributes.appendAssumeCapacity(.named(ctx, "collectorOp", value));
    if (opts.collectorOpB) |value| attributes.appendAssumeCapacity(.named(ctx, "collectorOpB", value));
    if (opts.aShift) |value| attributes.appendAssumeCapacity(.named(ctx, "aShift", value));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.mma.sp", .{
        .operands = .{ .variadic = &.{
            &.{matrixD},
            &.{matrixA},
            &.{matrixB},
            &.{idesc},
            &.{enableInputD},
            &.{sparseMetadata},
            if (opts.scaleInputD) |value| &.{value} else &.{},
            if (opts.disableOutputLane) |value| &.{value} else &.{},
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.mma.sp.block_scale`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_mma_sp_block_scale(
    ctx: *mlir.Context,
    matrixD: *const mlir.Value,
    matrixA: *const mlir.Value,
    matrixB: *const mlir.Value,
    idesc: *const mlir.Value,
    enableInputD: *const mlir.Value,
    sparseMetadata: *const mlir.Value,
    scaleA: *const mlir.Value,
    scaleB: *const mlir.Value,
    kind: *const mlir.Attribute,
    ctaGroup: *const mlir.Attribute,
    opts: struct {
        blockScale: ?*const mlir.Attribute = null,
        collectorOp: ?*const mlir.Attribute = null,
        collectorOpB: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    attributes.appendAssumeCapacity(.named(ctx, "ctaGroup", ctaGroup));
    if (opts.blockScale) |value| attributes.appendAssumeCapacity(.named(ctx, "blockScale", value));
    if (opts.collectorOp) |value| attributes.appendAssumeCapacity(.named(ctx, "collectorOp", value));
    if (opts.collectorOpB) |value| attributes.appendAssumeCapacity(.named(ctx, "collectorOpB", value));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.mma.sp.block_scale", .{
        .operands = .{ .flat = &.{ matrixD, matrixA, matrixB, idesc, enableInputD, sparseMetadata, scaleA, scaleB } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.mma.ws`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_mma_ws(
    ctx: *mlir.Context,
    matrixD: *const mlir.Value,
    matrixA: *const mlir.Value,
    matrixB: *const mlir.Value,
    idesc: *const mlir.Value,
    enableInputD: *const mlir.Value,
    kind: *const mlir.Attribute,
    opts: struct {
        zeroColMask: ?*const mlir.Value = null,
        collectorBBuffer: ?*const mlir.Attribute = null,
        collectorOp: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (opts.collectorBBuffer) |value| attributes.appendAssumeCapacity(.named(ctx, "collectorBBuffer", value));
    if (opts.collectorOp) |value| attributes.appendAssumeCapacity(.named(ctx, "collectorOp", value));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.mma.ws", .{
        .operands = .{ .flat = if (opts.zeroColMask) |value| &.{ matrixD, matrixA, matrixB, idesc, enableInputD, value } else &.{ matrixD, matrixA, matrixB, idesc, enableInputD } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.mma.ws.sp`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_mma_ws_sp(
    ctx: *mlir.Context,
    matrixD: *const mlir.Value,
    matrixA: *const mlir.Value,
    matrixB: *const mlir.Value,
    idesc: *const mlir.Value,
    enableInputD: *const mlir.Value,
    sparseMetadata: *const mlir.Value,
    kind: *const mlir.Attribute,
    opts: struct {
        zeroColMask: ?*const mlir.Value = null,
        collectorBBuffer: ?*const mlir.Attribute = null,
        collectorOp: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (opts.collectorBBuffer) |value| attributes.appendAssumeCapacity(.named(ctx, "collectorBBuffer", value));
    if (opts.collectorOp) |value| attributes.appendAssumeCapacity(.named(ctx, "collectorOp", value));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.mma.ws.sp", .{
        .operands = .{ .flat = if (opts.zeroColMask) |value| &.{ matrixD, matrixA, matrixB, idesc, enableInputD, sparseMetadata, value } else &.{ matrixD, matrixA, matrixB, idesc, enableInputD, sparseMetadata } },
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
pub fn tcgen05_relinquish_alloc_permit(
    ctx: *mlir.Context,
    opts: struct {
        group: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.group) |value| attributes.appendAssumeCapacity(.named(ctx, "group", value));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.relinquish_alloc_permit", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.shift`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_shift(
    ctx: *mlir.Context,
    taddr: *const mlir.Value,
    opts: struct {
        group: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.group) |value| attributes.appendAssumeCapacity(.named(ctx, "group", value));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.shift", .{
        .operands = .{ .flat = &.{taddr} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.tcgen05.st`. Result types are explicit; attributes use mlir.Attribute.
pub fn tcgen05_st(
    ctx: *mlir.Context,
    tmemAddr: *const mlir.Value,
    val: *const mlir.Value,
    shape: *const mlir.Attribute,
    opts: struct {
        offset: ?*const mlir.Value = null,
        unpack: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.unpack) |value| attributes.appendAssumeCapacity(.named(ctx, "unpack", value));
    attributes.appendAssumeCapacity(.named(ctx, "shape", shape));
    return mlir.Operation.make(ctx, "nvvm.tcgen05.st", .{
        .operands = .{ .flat = if (opts.offset) |value| &.{ tmemAddr, val, value } else &.{ tmemAddr, val } },
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
pub fn tensormap_replace(
    ctx: *mlir.Context,
    addr: *const mlir.Value,
    field: *const mlir.Attribute,
    opts: struct {
        new_value: ?*const mlir.Value = null,
        ord: ?*const mlir.Attribute = null,
        new_value_attr: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "field", field));
    if (opts.ord) |value| attributes.appendAssumeCapacity(.named(ctx, "ord", value));
    if (opts.new_value_attr) |value| attributes.appendAssumeCapacity(.named(ctx, "new_value_attr", value));
    return mlir.Operation.make(ctx, "nvvm.tensormap.replace", .{
        .operands = .{ .flat = if (opts.new_value) |value| &.{ addr, value } else &.{addr} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `nvvm.trace.mark`. Result types are explicit; attributes use mlir.Attribute.
pub fn trace_mark(
    ctx: *mlir.Context,
    eventType: *const mlir.Attribute,
    domain: *const mlir.Attribute,
    event: *const mlir.Attribute,
    opts: struct {
        payload: ?*const mlir.Value = null,
        predicate: ?*const mlir.Value = null,
        payloadDescriptor: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "eventType", eventType));
    attributes.appendAssumeCapacity(.named(ctx, "domain", domain));
    attributes.appendAssumeCapacity(.named(ctx, "event", event));
    if (opts.payloadDescriptor) |value| attributes.appendAssumeCapacity(.named(ctx, "payloadDescriptor", value));
    return mlir.Operation.make(ctx, "nvvm.trace.mark", .{
        .operands = .{ .variadic = &.{
            if (opts.payload) |value| &.{value} else &.{},
            if (opts.predicate) |value| &.{value} else &.{},
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
pub fn wgmma_mma_async(
    ctx: *mlir.Context,
    inouts: *const mlir.Value,
    descriptorA: *const mlir.Value,
    descriptorB: *const mlir.Value,
    results__type: *const mlir.Type,
    shape: *const mlir.Attribute,
    typeA: *const mlir.Attribute,
    typeB: *const mlir.Attribute,
    typeD: *const mlir.Attribute,
    scaleD: *const mlir.Attribute,
    scaleA: *const mlir.Attribute,
    scaleB: *const mlir.Attribute,
    layoutA: *const mlir.Attribute,
    layoutB: *const mlir.Attribute,
    opts: struct {
        satfinite: ?*const mlir.Attribute = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
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
    if (opts.satfinite) |value| attributes.appendAssumeCapacity(.named(ctx, "satfinite", value));
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
