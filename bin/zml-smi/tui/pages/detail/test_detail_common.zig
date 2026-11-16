const std = @import("std");
const common = @import("detail_common.zig");

const memoryPercent = common.memoryPercent;

test memoryPercent {
    const allocator = std.testing.allocator;
    try std.testing.expectEqualStrings("N/A", try memoryPercent(allocator, .{}));
    try std.testing.expectEqualStrings("N/A", try memoryPercent(allocator, .{
        .mem_total_bytes = 1024,
    }));
    try std.testing.expectEqualStrings("N/A", try memoryPercent(allocator, .{
        .mem_used_bytes = 0,
        .mem_total_bytes = 0,
    }));

    const idle = try memoryPercent(allocator, .{
        .mem_used_bytes = 0,
        .mem_total_bytes = 1024,
    });
    defer allocator.free(idle);
    try std.testing.expectEqualStrings("0%", idle);

    const full = try memoryPercent(allocator, .{
        .mem_used_bytes = std.math.maxInt(u64),
        .mem_total_bytes = std.math.maxInt(u64),
    });
    defer allocator.free(full);
    try std.testing.expectEqualStrings("100%", full);
}
