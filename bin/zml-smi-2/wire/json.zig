const std = @import("std");
const Snapshot = @import("zml-smi-2/snapshot").Snapshot;

pub fn write(writer: *std.Io.Writer, snapshot: Snapshot) !void {
    try std.json.Stringify.value(snapshot, .{ .emit_null_optional_fields = false }, writer);
    try writer.writeAll("\n");
}
