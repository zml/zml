const std = @import("std");
const memoryPercent = @import("presentation.zig").memoryPercent;

test memoryPercent {
    const allocator = std.testing.allocator;
    try std.testing.expectEqualStrings("N/A", try memoryPercent(allocator, null, null));
    try std.testing.expectEqualStrings("N/A", try memoryPercent(allocator, null, 1024));
    try std.testing.expectEqualStrings("N/A", try memoryPercent(allocator, 0, 0));

    const idle = try memoryPercent(allocator, 0, 1024);
    defer allocator.free(idle);
    try std.testing.expectEqualStrings("0%", idle);

    const max = std.math.maxInt(u64);
    const full = try memoryPercent(allocator, max, max);
    defer allocator.free(full);
    try std.testing.expectEqualStrings("100%", full);
}
