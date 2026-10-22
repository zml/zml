// Keep function definitions in ops.zig: LLVM reserves symbols starting with "llvm.".
const ops = @import("ops.zig");

pub const NVPTXAddressSpace = ops.NVPTXAddressSpace;
pub const pointerType = ops.pointerType;
pub const inttoptr = ops.inttoptr;
pub const bitcast = ops.bitcast;
pub const fmul = ops.fmul;
pub const LoadOpts = ops.LoadOpts;
pub const load = ops.load;
pub const StoreOpts = ops.StoreOpts;
pub const store = ops.store;
pub const call_intrinsic = ops.call_intrinsic;
pub const InlineAsmOpts = ops.InlineAsmOpts;
pub const inline_asm = ops.inline_asm;
pub const registerDialect = ops.registerDialect;
pub const dialect_namespace = ops.dialect_namespace;
pub const operation_names = ops.operation_names;
