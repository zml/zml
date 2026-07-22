const std = @import("std");
const zml = @import("zml");

const main = @import("main.zig");
const model_ = @import("model.zig");
const llm_ = @import("llm.zig");

const Zml_handler = main.Zml_handler;
const Model_handler = model_.Model_handler;
const Tokenizer = zml.tokenizer.Tokenizer;


pub fn decodeToken(tokenizer: zml.tokenizer.Tokenizer, token_id: u32, out: []u8) ![]const u8 {
    var decoder = try tokenizer.decoder();
    defer decoder.deinit();

    const chunk = try decoder.feedOne(token_id, out);
    const final_chunk = try decoder.finalize(out[chunk.len..]);
    return out[0 .. chunk.len + final_chunk.len];
}

pub fn escapeTokenText(text: []const u8, out: []u8) []const u8 {
    var len: usize = 0;
    for (text) |c| {
        const replacement = switch (c) {
            '\n' => "\\n",
            '\r' => "\\r",
            '\t' => "\\t",
            '\\' => "\\\\",
            else => null,
        };
        if (replacement) |rep| {
            if (len + rep.len > out.len) break;
            @memcpy(out[len..][0..rep.len], rep);
            len += rep.len;
        } else {
            if (len + 1 > out.len) break;
            out[len] = if (std.ascii.isControl(c)) '?' else c;
            len += 1;
        }
    }
    return out[0..len];
}


fn codepointInExcludedRanges(codepoint: u21) bool {
    for (excluded_unicode_ranges) |range| {
        if (range.start <= codepoint and codepoint <= range.end) return true;
    }
    return false;
}

fn isUnicodeWhitespace(codepoint: u21) bool {
    return switch (codepoint) {
        0x0009...0x000D,
        0x0020,
        0x0085,
        0x00A0,
        0x1680,
        0x2000...0x200A,
        0x2028,
        0x2029,
        0x202F,
        0x205F,
        0x3000,
        => true,
        else => false,
    };
}

const UnicodeRange = struct {
    start: u21,
    end: u21,
};

const excluded_unicode_ranges = [_]UnicodeRange{
    .{ .start = 0x3400, .end = 0x4DBF },
    .{ .start = 0x4E00, .end = 0x9FFF },
    .{ .start = 0x20000, .end = 0x2A6DF },
    .{ .start = 0x2A700, .end = 0x2B73F },
    .{ .start = 0x2B740, .end = 0x2B81F },
    .{ .start = 0x2B820, .end = 0x2CEAF },
    .{ .start = 0x2CEB0, .end = 0x2EBEF },
    .{ .start = 0x30000, .end = 0x3134F },
    .{ .start = 0xF900, .end = 0xFAFF },
    .{ .start = 0x3040, .end = 0x309F },
    .{ .start = 0x30A0, .end = 0x30FF },
    .{ .start = 0x31F0, .end = 0x31FF },
    .{ .start = 0xFF66, .end = 0xFF9F },
    .{ .start = 0x1100, .end = 0x11FF },
    .{ .start = 0x3130, .end = 0x318F },
    .{ .start = 0xAC00, .end = 0xD7AF },
    .{ .start = 0xA960, .end = 0xA97F },
    .{ .start = 0xD7B0, .end = 0xD7FF },
    .{ .start = 0x0400, .end = 0x052F },
    .{ .start = 0x1C80, .end = 0x1C8F },
    .{ .start = 0x2DE0, .end = 0x2DFF },
    .{ .start = 0xA640, .end = 0xA69F },
    .{ .start = 0x0600, .end = 0x06FF },
    .{ .start = 0x0750, .end = 0x077F },
    .{ .start = 0x0870, .end = 0x089F },
    .{ .start = 0x08A0, .end = 0x08FF },
    .{ .start = 0xFB50, .end = 0xFDFF },
    .{ .start = 0xFE70, .end = 0xFEFF },
    .{ .start = 0x0590, .end = 0x05FF },
    .{ .start = 0x0900, .end = 0x097F },
    .{ .start = 0xA8E0, .end = 0xA8FF },
    .{ .start = 0x0980, .end = 0x09FF },
    .{ .start = 0x0A00, .end = 0x0A7F },
    .{ .start = 0x0A80, .end = 0x0AFF },
    .{ .start = 0x0B80, .end = 0x0BFF },
    .{ .start = 0x0C00, .end = 0x0C7F },
    .{ .start = 0x0C80, .end = 0x0CFF },
    .{ .start = 0x0D00, .end = 0x0D7F },
    .{ .start = 0x0D80, .end = 0x0DFF },
    .{ .start = 0x0E00, .end = 0x0E7F },
    .{ .start = 0x0E80, .end = 0x0EFF },
    .{ .start = 0x1780, .end = 0x17FF },
    .{ .start = 0x19E0, .end = 0x19FF },
    .{ .start = 0x1000, .end = 0x109F },
    .{ .start = 0xA9E0, .end = 0xA9FF },
    .{ .start = 0xAA60, .end = 0xAA7F },
    .{ .start = 0x0F00, .end = 0x0FFF },
    .{ .start = 0x1800, .end = 0x18AF },
    .{ .start = 0x0530, .end = 0x058F },
    .{ .start = 0x10A0, .end = 0x10FF },
    .{ .start = 0x1C90, .end = 0x1CBF },
    .{ .start = 0x2D00, .end = 0x2D2F },
    .{ .start = 0x1200, .end = 0x137F },
    .{ .start = 0x1380, .end = 0x139F },
    .{ .start = 0x2D80, .end = 0x2DDF },
    .{ .start = 0xAB00, .end = 0xAB2F },
};
