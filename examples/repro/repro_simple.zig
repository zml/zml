//! Compiler segfault: recursive comptime `@typeInfo` on self-referential struct
//!
//!   OK  quota <= 1849: clean "evaluation exceeded N backwards branches" error
//!   BAD quota >= 1850: Segmentation fault (stack overflow in compiler)
//!
//!   OK  https://codeberg.org/ziglang/zig/commit/f16eb18ce8c24
//!   BAD https://codeberg.org/ziglang/zig/commit/fd2718f82ab70
//!
//! `zig build-exe repro_simple.zig`  →  Segmentation fault
const A = struct { a: *const A };

fn walk(comptime H: type) void {
    @setEvalBranchQuota(2000);
    switch (@typeInfo(H)) {
        .@"struct" => |i| inline for (i.fields) |f| walk(f.type),
        .pointer => |i| walk(i.child),
        else => {},
    }
}

comptime {
    walk(A);
}
pub fn main() void {}
