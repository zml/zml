// Keep function definitions in ops.zig: LLVM reserves symbols starting with "llvm.".
const ops = @import("ops.zig");

pub const NVPTXAddressSpace = ops.NVPTXAddressSpace;
pub const pointerType = ops.pointerType;
pub const inttoptr = ops.inttoptr;
pub const bitcast = ops.bitcast;
pub const fmul = ops.fmul;
pub const LoadOptions = ops.LoadOptions;
pub const load = ops.load;
pub const StoreOptions = ops.StoreOptions;
pub const store = ops.store;
pub const call_intrinsic = ops.call_intrinsic;
pub const InlineAsmOptions = ops.InlineAsmOptions;
pub const inline_asm = ops.inline_asm;
