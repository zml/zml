const std = @import("std");
const model = @import("zml-smi/model");

/// Every device with its latest sample, then the merged processes.
pub fn write(writer: *std.Io.Writer, frame: model.Frame) !void {
    var json: std.json.Stringify = .{ .writer = writer };
    try json.beginObject();
    try json.objectField("devices");
    try json.beginArray();
    for (frame.devices, frame.samples) |device, sample| {
        try json.write(model.Row{ .device = device, .sample = sample });
    }
    try json.endArray();

    try json.objectField("processes");
    try json.write(frame.processes);
    try json.endObject();
    try writer.writeAll("\n");
}
