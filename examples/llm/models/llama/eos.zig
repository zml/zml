const std = @import("std");

pub fn contains(token_ids: anytype, token_id: u32) bool {
    return switch (token_ids.value) {
        .int => |eos| token_id == eos,
        .ints => |eos_list| for (eos_list) |eos| {
            if (token_id == eos) break true;
        } else false,
    };
}

const scalarEosTokenId: u32 = 128001;
const eotTokenId: u32 = 128008;
const pythonTagTokenId: u32 = 128009;
const nonEosTokenId: u32 = 42;

const TestTokenIds = union(enum) {
    int: u32,
    ints: []u32,
};

const TestConfigTokenIds = struct {
    value: TestTokenIds,
};

test "contains accepts scalar EOS token ids" {
    const token_ids: TestConfigTokenIds = .{ .value = .{ .int = scalarEosTokenId } };

    try std.testing.expect(contains(token_ids, scalarEosTokenId));
    try std.testing.expect(!contains(token_ids, nonEosTokenId));
}

test "contains accepts every configured EOS token id" {
    var eos_token_ids = [_]u32{ scalarEosTokenId, eotTokenId, pythonTagTokenId };
    const token_ids: TestConfigTokenIds = .{ .value = .{ .ints = &eos_token_ids } };

    for (eos_token_ids) |eos_token_id| {
        try std.testing.expect(contains(token_ids, eos_token_id));
    }
    try std.testing.expect(!contains(token_ids, nonEosTokenId));
}
