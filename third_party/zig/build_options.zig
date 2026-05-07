const std = @import("std");

pub const IoMode = enum { threaded, evented };
pub const ValueInterpretMode = enum { direct, by_name };
pub const DevEnv = enum {
    bootstrap,
    core,
    full,
    c_source,
    ast_gen,
    sema,
    @"aarch64-linux",
    cbe,
    @"powerpc-linux",
    @"riscv64-linux",
    spirv,
    wasm,
    @"x86_64-linux",
};

pub const mem_leak_frames: u32 = 4;
pub const skip_non_native: bool = false;
pub const have_llvm: bool = true;
pub const have_llvm_clang: bool = false;
pub const have_llvm_ar: bool = false;
pub const llvm_has_m68k: bool = false;
pub const llvm_has_csky: bool = false;
pub const llvm_has_arc: bool = false;
pub const llvm_has_xtensa: bool = false;
pub const debug_gpa: bool = false;
pub const dev: DevEnv = .full;
pub const io_mode: IoMode = .threaded;
pub const value_interpret_mode: ValueInterpretMode = .direct;

pub const version: []const u8 = "0.16.0";
pub const semver: std.SemanticVersion = .{
    .major = 0,
    .minor = 16,
    .patch = 0,
};

pub const enable_debug_extensions: bool = true;
pub const enable_logging: bool = true;
pub const enable_link_snapshots: bool = false;
pub const enable_tracy: bool = false;
pub const enable_tracy_callstack: bool = false;
pub const enable_tracy_allocation: bool = false;
pub const tracy_callstack_depth: u32 = 10;
pub const value_tracing: bool = false;

// Compatibility toggles checked by @hasDecl in src/dev.zig.
pub const only_c: bool = false;
pub const only_core_functionality: bool = false;
