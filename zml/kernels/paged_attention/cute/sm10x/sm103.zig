const std = @import("std");
const mlir = @import("mlir");
const llvm = @import("mlir/dialects/llvm");

const cute = @import("../../../../zml.zig").kernel.cute;

pub const Scores = struct {
    values: [64]cute.Value,
    maximum: cute.Value,
};

// SM103 adds a per-TMEM-lane reduction across the loaded columns. This is
// useful for query-major scores; swapped decode reduces across lanes instead.
pub fn loadScores(b: *cute.Builder, address: cute.Value) Scores {
    const asm_parts = comptime blk: {
        @setEvalBranchQuota(20000);
        var registers: []const u8 = "";
        var constraints: []const u8 = "";
        for (0..65) |i| {
            if (i < 64) registers = registers ++ (if (i == 0) "" else ", ") ++ std.fmt.comptimePrint("${d}", .{i});
            constraints = constraints ++ (if (i == 0) "" else ",") ++ "=f";
        }
        break :blk .{
            "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32 {" ++ registers ++ "}, $64, [$65];",
            constraints ++ ",r",
        };
    };
    const result = b.emit(llvm.inline_asm(b.ctx, asm_parts[0], &.{address.inner}, .{ .result_type = llvm.structType(b.ctx, &@as([65]*const mlir.Type, @splat(.float(b.ctx, .f32)))), .constraints = asm_parts[1], .has_side_effects = true }, b.loc()));
    var scores: Scores = undefined;
    for (0..65) |i| {
        const value = b.emit(llvm.extractvalue(b.ctx, result.inner, cute.DType.f32.toMlir(b.ctx), &.{@intCast(i)}, b.loc()));
        if (i < 64) scores.values[i] = value else scores.maximum = value;
    }
    b.fenceTmemLoad();
    return scores;
}
