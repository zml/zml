const std = @import("std");

/// Check the architecture fields used by the native equations. Training and
/// generation metadata may vary without changing checkpoint compatibility.
pub fn validate(a: std.mem.Allocator, model: []const u8, codec: []const u8) !void {
    const expected = try std.json.parseFromSlice(std.json.Value, a, @embedFile("supported_config.json"), .{});
    defer expected.deinit();
    const actual_model = try std.json.parseFromSlice(std.json.Value, a, model, .{});
    defer actual_model.deinit();
    const actual_codec = try std.json.parseFromSlice(std.json.Value, a, codec, .{});
    defer actual_codec.deinit();
    if (!matches(expected.value.object.get("model").?, actual_model.value) or !matches(expected.value.object.get("codec").?, actual_codec.value)) return error.UnsupportedBreezeCheckpoint;
    if (actual_model.value.object.get("text_encoder_feature_layer_idx")) |v| {
        if (v != .integer or v.integer != -1) return error.UnsupportedTextEncoderFeatures;
    }
}
fn matches(expected: std.json.Value, actual: std.json.Value) bool {
    return switch (expected) {
        .object => |o| blk: {
            if (actual != .object) break :blk false;
            var it = o.iterator();
            while (it.next()) |entry| {
                const v = actual.object.get(entry.key_ptr.*) orelse break :blk false;
                if (!matches(entry.value_ptr.*, v)) break :blk false;
            }
            break :blk true;
        },
        .array => |v| blk: {
            if (actual != .array or actual.array.items.len != v.items.len) break :blk false;
            for (v.items, actual.array.items) |e, x| if (!matches(e, x)) break :blk false;
            break :blk true;
        },
        .integer => |v| switch (actual) {
            .integer => |x| x == v,
            .float => |x| x == @as(f64, @floatFromInt(v)),
            else => false,
        },
        .float => |v| switch (actual) {
            .float => |x| x == v,
            .integer => |x| v == @as(f64, @floatFromInt(x)),
            else => false,
        },
        .string => |v| actual == .string and std.mem.eql(u8, v, actual.string),
        .bool => |v| actual == .bool and actual.bool == v,
        .null => actual == .null,
        else => false,
    };
}
test "architecture checks ignore extra metadata and reject changed equations" {
    const a = std.testing.allocator;
    const expected = try std.json.parseFromSlice(std.json.Value, a, "{\"layers\":2,\"eps\":0.00001}", .{});
    defer expected.deinit();
    const valid = try std.json.parseFromSlice(std.json.Value, a, "{\"layers\":2.0,\"eps\":0.00001,\"description\":\"fine tune\"}", .{});
    defer valid.deinit();
    const wrong = try std.json.parseFromSlice(std.json.Value, a, "{\"layers\":2,\"eps\":0.000001}", .{});
    defer wrong.deinit();
    try std.testing.expect(matches(expected.value, valid.value));
    try std.testing.expect(!matches(expected.value, wrong.value));
}
