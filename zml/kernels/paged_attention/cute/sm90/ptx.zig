//! Hopper instructions the sm90 attention kernel needs and the CuTe builder does not wrap: warpgroup MMA and register
//! packing. Adapted from the flash attention proof of concept. Shared memory addresses are i32 offsets in the shared window.
const std = @import("std");

const builder = @import("dsl/cute/builder");
const mlir = @import("mlir");
const vector = @import("mlir/dialects/vector");

const B = builder.Builder;
const V = builder.Value;
const DType = builder.DType;

fn separator(index: usize) []const u8 {
    if (index == 0) {
        return "";
    }
    return ", ";
}

fn print(b: *B, comptime format: []const u8, args: anytype) []const u8 {
    return std.fmt.allocPrint(b.arena.allocator(), format, args) catch @panic("OOM");
}

fn registerClass(dtype: DType) []const u8 {
    if (dtype == .f32) {
        return "f";
    }
    return "r";
}

/// The builder's loop-carried values are limited to 64, so accumulators travel through loops as one vector.
pub fn packVector(b: *B, values: []const V) V {
    const elements = b.arena.allocator().alloc(*const mlir.Value, values.len) catch @panic("OOM");
    for (values, elements) |value, *element| {
        element.* = value.inner;
    }
    const type_ = mlir.Type.vector(&[_]i64{@intCast(values.len)}, values[0].type_());
    return b.emit(vector.from_elements(b.ctx, elements, type_, b.loc()));
}

pub fn unpackVector(b: *B, comptime n: usize, packed_values: V, dtype: DType) [n]V {
    var values: [n]V = undefined;
    for (&values, 0..) |*value, lane| {
        value.* = b.vectorExtract(packed_values, @intCast(lane), dtype);
    }
    return values;
}

pub fn wgmmaFence(b: *B) void {
    builder.asmVoid(b, "wgmma.fence.sync.aligned;", &.{}, "~{memory}");
}

pub fn wgmmaCommit(b: *B) void {
    builder.asmVoid(b, "wgmma.commit_group.sync.aligned;", &.{}, "~{memory}");
}

pub fn wgmmaWait(b: *B, comptime pending: u32) void {
    builder.asmVoid(b, print(b, "wgmma.wait_group.sync.aligned {d};", .{pending}), &.{}, "~{memory}");
}

/// Makes writes by cp.async (generic proxy) visible to wgmma (async proxy).
pub fn asyncProxyFence(b: *B) void {
    builder.asmVoid(b, "fence.proxy.async.shared::cta;", &.{}, "~{memory}");
}

/// An empty asm that reads and writes every register, so the compiler cannot move arithmetic on accumulators across wgmma.
pub fn fenceOperands(b: *B, comptime dtype: DType, values: []V) void {
    const register_class = registerClass(dtype);
    var constraints: std.Io.Writer.Allocating = .init(b.arena.allocator());
    for (values) |_| {
        constraints.writer.print("={s},", .{register_class}) catch @panic("OOM");
    }
    for (values, 0..) |_, i| {
        constraints.writer.print("{d},", .{i}) catch @panic("OOM");
    }
    constraints.writer.print("~{{memory}}", .{}) catch @panic("OOM");
    const fenced = builder.asmValuesSlice(b, values.len, "", values, dtype, constraints.written(), true);
    @memcpy(values, fenced);
}

/// Descriptor of a 128-byte-swizzled operand: address >> 4 in bits 0-13, leading byte offset >> 4 in 16-29, stride byte
/// offset >> 4 in 32-45, swizzle mode 1 in 62-63. Only the low word varies at run time.
pub const MatrixDescriptor = struct {
    low: V,
    high: u32,

    pub fn advance(self: MatrixDescriptor, units: i32) MatrixDescriptor {
        return .{ .low = self.low.add(units), .high = self.high };
    }
};

pub fn matrixDescriptor(b: *B, address: V, leading_byte_offset: u32, stride_byte_offset: u32) MatrixDescriptor {
    const swizzle_128: u32 = 1;
    const fixed_low: i32 = @intCast((leading_byte_offset >> 4) << 16);
    return .{
        .low = builder.shflSync(b, address.bitAnd(0x3FFFF).shrLogical(4).bitOr(fixed_low), 0),
        .high = (stride_byte_offset >> 4) | (swizzle_128 << 30),
    };
}

/// wgmma m64n{n}k16, f32 accumulators, both operands in shared memory and K-major.
pub fn wgmmaSharedShared(b: *B, comptime dtype: DType, comptime n: u32, accumulators: *[n / 2]V, descriptor_a: MatrixDescriptor, descriptor_b: MatrixDescriptor, comptime accumulate: bool) void {
    const count = n / 2;
    var arguments: [count + 2]V = undefined;
    @memcpy(arguments[0..count], accumulators);
    arguments[count] = descriptor_a.low;
    arguments[count + 1] = descriptor_b.low;
    const text = wgmmaText(b, @tagName(dtype), n, false, accumulate, descriptor_a.high, descriptor_b.high);
    const results = builder.asmValuesSlice(b, count, text, &arguments, .f32, wgmmaConstraints(b, count, "r,r"), true);
    @memcpy(accumulators, results);
}

/// wgmma m64n{n}k16 with A in registers (four 16-bit-pair words) and a transposed (MN-major) B in shared memory.
pub fn wgmmaRegistersShared(b: *B, comptime dtype: DType, comptime n: u32, accumulators: *[n / 2]V, a: [4]V, descriptor_b: MatrixDescriptor) void {
    const count = n / 2;
    var arguments: [count + 5]V = undefined;
    @memcpy(arguments[0..count], accumulators);
    @memcpy(arguments[count..][0..4], &a);
    arguments[count + 4] = descriptor_b.low;
    const text = wgmmaText(b, @tagName(dtype), n, true, true, 0, descriptor_b.high);
    const results = builder.asmValuesSlice(b, count, text, &arguments, .f32, wgmmaConstraints(b, count, "r,r,r,r,r"), true);
    @memcpy(accumulators, results);
}

fn wgmmaConstraints(b: *B, count: usize, operands: []const u8) []const u8 {
    var constraints: std.Io.Writer.Allocating = .init(b.arena.allocator());
    for (0..count) |_| {
        constraints.writer.print("=f,", .{}) catch @panic("OOM");
    }
    for (0..count) |i| {
        constraints.writer.print("{d},", .{i}) catch @panic("OOM");
    }
    constraints.writer.print("{s},~{{memory}}", .{operands}) catch @panic("OOM");
    return constraints.written();
}

fn wgmmaText(b: *B, type_name: []const u8, comptime n: u32, comptime a_in_registers: bool, comptime accumulate: bool, high_a: u32, high_b: u32) []const u8 {
    const count = n / 2;
    var text: std.Io.Writer.Allocating = .init(b.arena.allocator());
    const writer = &text.writer;
    writer.print("{{\n.reg .pred p;\n.reg .b64 descriptor_a;\n.reg .b64 descriptor_b;\nsetp.ne.b32 p, {d}, 0;\n", .{@intFromBool(accumulate)}) catch @panic("OOM");
    if (a_in_registers) {
        writer.print("mov.b64 descriptor_b, {{${d}, {d}}};\n", .{ 2 * count + 4, high_b }) catch @panic("OOM");
    } else {
        writer.print("mov.b64 descriptor_a, {{${d}, {d}}};\nmov.b64 descriptor_b, {{${d}, {d}}};\n", .{ 2 * count, high_a, 2 * count + 1, high_b }) catch @panic("OOM");
    }
    writer.print("wgmma.mma_async.sync.aligned.m64n{d}k16.f32.{s}.{s} {{", .{ n, type_name, type_name }) catch @panic("OOM");
    for (0..count) |i| {
        writer.print("{s}${d}", .{ separator(i), i }) catch @panic("OOM");
    }
    if (a_in_registers) {
        const a_first = 2 * count;
        writer.print("}}, {{${d}, ${d}, ${d}, ${d}}}, descriptor_b, p, 1, 1, 1;\n}}", .{ a_first, a_first + 1, a_first + 2, a_first + 3 }) catch @panic("OOM");
    } else {
        writer.print("}}, descriptor_a, descriptor_b, p, 1, 1, 0, 0;\n}}", .{}) catch @panic("OOM");
    }
    return text.written();
}

/// Two f32 into one register of two 16-bit values, `low_element` in bits 0-15.
pub fn packHalves(b: *B, comptime dtype: DType, low_element: V, high_element: V) V {
    return builder.asmValue(b, "cvt.rn." ++ @tagName(dtype) ++ "x2.f32 $0, $1, $2;", &.{ high_element, low_element }, .i32, "=r,f,f", false);
}
