const std = @import("std");

const zml = @import("zml");
const chat_template = @import("kimi_k3/chat_template.zig");
const kimi_tokenizer = @import("kimi_k3/tokenizer.zig");

pub const std_options: std.Options = .{ .log_level = .info };

const Args = struct {
    tokenizer: []const u8,
    fixture: []const u8,

    pub const help =
        \\Use kimi_k3_tokenizer_tests --tokenizer=<tokenizer.json> --fixture=<tokenizer-reference.json>
        \\
        \\Compare converted ZML tokenization and simple XTML chat formatting to Moonshot.
        \\
    ;
};

const TextCase = struct {
    name: []const u8,
    text: []const u8,
    token_ids: []const u32,
    decoded: []const u8,
};

const StructuralCase = struct {
    text: []const u8,
    token_id: u32,
};

const ChatCase = struct {
    name: []const u8,
    prompt: []const u8,
    rendered: []const u8,
    token_ids: []const u32,
};

const Fixture = struct {
    vocab_size: u32,
    bos_token_id: u32,
    eos_token_id: u32,
    end_of_message_token_id: u32,
    ordinary: []const TextCase,
    fuzz: []const TextCase,
    structural: []const StructuralCase,
    first_turn: []const ChatCase,
    continuation: []const ChatCase,
};

fn expectTokens(label: []const u8, expected: []const u32, actual: []const u32) !void {
    if (!std.mem.eql(u32, expected, actual)) {
        std.log.err("Kimi K3 tokenizer mismatch for {s}: expected={any} actual={any}", .{ label, expected, actual });
        return error.KimiK3TokenizerMismatch;
    }
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);

    const fixture_bytes = try std.Io.Dir.cwd().readFileAlloc(io, args.fixture, allocator, .unlimited);
    defer allocator.free(fixture_bytes);
    var fixture = try std.json.parseFromSlice(Fixture, allocator, fixture_bytes, .{ .ignore_unknown_fields = true });
    defer fixture.deinit();
    if (fixture.value.vocab_size != 163840 or fixture.value.bos_token_id != 163584 or
        fixture.value.eos_token_id != 163585 or fixture.value.end_of_message_token_id != 163586)
    {
        return error.InvalidKimiK3TokenizerFixture;
    }

    var tokenizer = try zml.tokenizer.Tokenizer.fromFile(allocator, io, args.tokenizer);
    defer tokenizer.deinit();

    for (fixture.value.structural) |case| {
        if (tokenizer.tokenId(case.text) != case.token_id) return error.KimiK3StructuralTokenMismatch;
    }

    for (fixture.value.ordinary) |case| {
        const actual = try kimi_tokenizer.encodeText(allocator, tokenizer, case.text);
        defer allocator.free(actual);
        try expectTokens(case.name, case.token_ids, actual);

        var decoder = try tokenizer.decoder();
        defer decoder.deinit();
        var decoded = try decoder.decodeAlloc(allocator, actual);
        defer decoded.deinit(allocator);
        if (!std.mem.eql(u8, case.decoded, decoded.items)) return error.KimiK3TokenizerDecodeMismatch;
        std.log.info("KIMI_K3_TOKENIZER_TEXT_PASS case={s} tokens={}", .{ case.name, actual.len });
    }

    for (fixture.value.fuzz) |case| {
        const actual = try kimi_tokenizer.encodeText(allocator, tokenizer, case.text);
        defer allocator.free(actual);
        try expectTokens(case.name, case.token_ids, actual);

        var decoder = try tokenizer.decoder();
        defer decoder.deinit();
        var decoded = try decoder.decodeAlloc(allocator, actual);
        defer decoded.deinit(allocator);
        if (!std.mem.eql(u8, case.decoded, decoded.items)) return error.KimiK3TokenizerDecodeMismatch;
    }
    std.log.info("KIMI_K3_TOKENIZER_FUZZ_PASS cases={}", .{fixture.value.fuzz.len});

    for (fixture.value.first_turn) |case| {
        const actual = try chat_template.tokenizePrompt(allocator, tokenizer, case.prompt);
        defer allocator.free(actual);
        try expectTokens(case.name, case.token_ids, actual);
        std.log.info("KIMI_K3_TOKENIZER_CHAT_PASS phase=first case={s} tokens={}", .{ case.name, actual.len });
    }
    for (fixture.value.continuation) |case| {
        const actual = try chat_template.tokenizeTurn(allocator, tokenizer, case.prompt);
        defer allocator.free(actual);
        try expectTokens(case.name, case.token_ids, actual);
        std.log.info("KIMI_K3_TOKENIZER_CHAT_PASS phase=continuation case={s} tokens={}", .{ case.name, actual.len });
    }

    const stdout = std.Io.File.stdout();
    var buffer: [512]u8 = undefined;
    var writer = stdout.writer(io, &buffer);
    try writer.interface.print(
        "KIMI_K3_TOKENIZER_ALL_PASS ordinary={} fuzz={} structural={} first_turn={} continuation={}\n",
        .{ fixture.value.ordinary.len, fixture.value.fuzz.len, fixture.value.structural.len, fixture.value.first_turn.len, fixture.value.continuation.len },
    );
    try writer.interface.flush();
}
