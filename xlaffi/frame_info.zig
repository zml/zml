const std = @import("std");
const ffi = @import("ffi.zig");

pub fn printCallFrameInfo(frame: *ffi.CallFrame, writer: anytype) !void {
    try writer.print("=== Call Frame Info ===\n", .{});

    // Execution stage
    try writer.print("Stage: {s}\n", .{@tagName(frame.stage)});

    // Arguments
    try writer.print("\nArguments ({} args):\n", .{frame.args.size});
    for (0..@intCast(frame.args.size)) |i| {
        const arg_type = frame.args.types[i];
        try writer.print("  [{d}] Type: {s}\n", .{ i, @tagName(arg_type) });
    }

    // Returns
    try writer.print("\nReturns ({} rets):\n", .{frame.rets.size});
    for (0..@intCast(frame.rets.size)) |i| {
        const ret_type = frame.rets.types[i];
        try writer.print("  [{d}] Type: {s}\n", .{ i, @tagName(ret_type) });
    }

    // Attributes
    try writer.print("\nAttributes ({} attrs):\n", .{frame.attrs.size});
    for (0..@intCast(frame.attrs.size)) |i| {
        const attr_type = frame.attrs.types[i];
        const attr_name = frame.attrs.getName(i);
        try writer.print("  [{d}] Name: {s}, Type: {s}\n", .{
            i,
            attr_name,
            @tagName(attr_type),
        });
    }

    // Context info
    try writer.print("\nContext: {?*}\n", .{frame.ctx});
    try writer.print("Future: {?*}\n", .{frame.future});
}

pub fn getBufferInfo(buffer: *ffi.Buffer, writer: anytype) !void {
    try writer.print("=== Buffer Info ===\n", .{});
    try writer.print("DataType: {s}\n", .{@tagName(buffer.dtype)});
    try writer.print("Rank: {d}\n", .{buffer.rank});

    if (buffer.rank > 0) {
        try writer.print("Dimensions: [", .{});
        for (0..@intCast(buffer.rank)) |i| {
            if (i > 0) try writer.print(", ", .{});
            try writer.print("{d}", .{buffer._dims[i]});
        }
        try writer.print("]\n", .{});
    }

    try writer.print("Data pointer: {?*}\n", .{buffer.data});
}
