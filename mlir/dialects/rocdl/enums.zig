// Generated from LLVM ROCDL TableGen records; see REGENERATING.md.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// ROCDL buffer out-of-bounds mode
pub const BufferOOBMode = enum(u32) {
    Any = 0,
    Relaxed = 1,
    Strict = 2,
};

/// gfx12 atomic cache policy bits
pub const Gfx12AtomicCachePolicy = enum(u32) {
    none = 0,
    keep_return = 1,
    nt = 2,
    cascade = 4,
    scope_se = 8,
    scope_dev = 16,
    scope_sys = 24,
    nv = 32,
    volatile_op = 2147483648,
    _,

    pub fn combine(values: []const Gfx12AtomicCachePolicy) Gfx12AtomicCachePolicy {
        var bits: u32 = 0;
        for (values) |value| bits |= @intFromEnum(value);
        return @enumFromInt(bits);
    }

    pub fn contains(self: Gfx12AtomicCachePolicy, value: Gfx12AtomicCachePolicy) bool {
        return @intFromEnum(self) & @intFromEnum(value) == @intFromEnum(value);
    }
};

/// gfx12 cache policy bits
pub const Gfx12CachePolicy = enum(u32) {
    none = 0,
    nt = 1,
    ht = 2,
    lu = 3,
    nt_rt = 4,
    rt_nt = 5,
    nt_ht = 6,
    nt_wb = 7,
    scope_se = 8,
    scope_dev = 16,
    scope_sys = 24,
    nv = 32,
    swz = 64,
    scal = 2048,
    volatile_op = 2147483648,
    _,

    pub fn combine(values: []const Gfx12CachePolicy) Gfx12CachePolicy {
        var bits: u32 = 0;
        for (values) |value| bits |= @intFromEnum(value);
        return @enumFromInt(bits);
    }

    pub fn contains(self: Gfx12CachePolicy, value: Gfx12CachePolicy) bool {
        return @intFromEnum(self) & @intFromEnum(value) == @intFromEnum(value);
    }
};

/// gfx942 buffer cache policy bits
pub const Gfx942CachePolicy = enum(u32) {
    none = 0,
    sc0 = 1,
    nt = 2,
    swz = 8,
    sc1 = 16,
    volatile_op = 2147483648,
    _,

    pub fn combine(values: []const Gfx942CachePolicy) Gfx942CachePolicy {
        var bits: u32 = 0;
        for (values) |value| bits |= @intFromEnum(value);
        return @enumFromInt(bits);
    }

    pub fn contains(self: Gfx942CachePolicy, value: Gfx942CachePolicy) bool {
        return @intFromEnum(self) & @intFromEnum(value) == @intFromEnum(value);
    }
};

/// negation modifier bitfield for gfx94x double-precision MFMA
pub const MFMANegModifier = enum(u32) {
    none = 0,
    neg_a = 1,
    neg_b = 2,
    neg_c = 4,
    _,

    pub fn combine(values: []const MFMANegModifier) MFMANegModifier {
        var bits: u32 = 0;
        for (values) |value| bits |= @intFromEnum(value);
        return @enumFromInt(bits);
    }

    pub fn contains(self: MFMANegModifier, value: MFMANegModifier) bool {
        return @intFromEnum(self) & @intFromEnum(value) == @intFromEnum(value);
    }
};

/// permutations of the lanes storing B in an MFMA
pub const MFMAPermB = enum(u32) {
    none = 0,
    bcast_first_32 = 1,
    bcast_second_32 = 2,
    rotate_16_right = 3,
    bcast_first_16 = 4,
    bcast_second_16 = 5,
    bcast_third_16 = 6,
    bcast_fourth_16 = 7,
};

/// matrix operand formats selected by scaled MFMA/WMMA format fields
pub const MatrixFormat = enum(u32) {
    fp8_e4m3 = 0,
    fp8_e5m2 = 1,
    fp6_e2m3 = 2,
    fp6_e3m2 = 3,
    fp4_e2m1 = 4,
};

/// pre-gfx12 buffer cache policy bits
pub const PreGfx12CachePolicy = enum(u32) {
    none = 0,
    glc = 1,
    slc = 2,
    dlc = 4,
    swz = 8,
    scc = 16,
    all = 23,
    volatile_op = 2147483648,
    _,

    pub fn combine(values: []const PreGfx12CachePolicy) PreGfx12CachePolicy {
        var bits: u32 = 0;
        for (values) |value| bits |= @intFromEnum(value);
        return @enumFromInt(bits);
    }

    pub fn contains(self: PreGfx12CachePolicy, value: PreGfx12CachePolicy) bool {
        return @intFromEnum(self) & @intFromEnum(value) == @intFromEnum(value);
    }
};

/// instruction type mask for scheduling barriers
pub const SchedGroupMask = enum(u32) {
    none = 0,
    non_mem_non_sideeffect = 1,
    valu = 2,
    salu = 4,
    mfma_wmma = 8,
    all_vmem = 16,
    vmem_read = 32,
    vmem_write = 64,
    all_ds = 128,
    ds_read = 256,
    ds_write = 512,
    transcendental = 1024,
    ldsdma = 2048,
    all = 4095,
    _,

    pub fn combine(values: []const SchedGroupMask) SchedGroupMask {
        var bits: u32 = 0;
        for (values) |value| bits |= @intFromEnum(value);
        return @enumFromInt(bits);
    }

    pub fn contains(self: SchedGroupMask, value: SchedGroupMask) bool {
        return @intFromEnum(self) & @intFromEnum(value) == @intFromEnum(value);
    }
};

/// WMMA C operand modifiers
pub const WMMACModifier = enum(u16) {
    none = 0,
    neg = 1,
    abs = 2,
    neg_abs = 3,
    _,

    pub fn combine(values: []const WMMACModifier) WMMACModifier {
        var bits: u16 = 0;
        for (values) |value| bits |= @intFromEnum(value);
        return @enumFromInt(bits);
    }

    pub fn contains(self: WMMACModifier, value: WMMACModifier) bool {
        return @intFromEnum(self) & @intFromEnum(value) == @intFromEnum(value);
    }
};

/// matrix scale row selector
pub const WMMAMatrixScale = enum(u32) {
    row0 = 0,
    row1 = 1,
};

/// matrix scale exponent formats
pub const WMMAMatrixScaleFormat = enum(u32) {
    e8 = 0,
    e5m3 = 1,
    e4m3 = 2,
};
