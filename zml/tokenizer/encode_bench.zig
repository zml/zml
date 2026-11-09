//! CPU encode timing with an independent full-token-ID reference.
const std = @import("std");

const tokenizer_lib = @import("zml/tokenizer");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    var tokenizer_path: ?[]const u8 = null;
    var text_path: ?[]const u8 = null;
    var expected_path: ?[]const u8 = null;
    var iterations: usize = 100;
    var warmups: usize = 10;
    var args = init.minimal.args.iterate();
    _ = args.next();
    while (args.next()) |arg| {
        const eq = std.mem.indexOfScalar(u8, arg, '=') orelse return error.InvalidArgument;
        const key = arg[0..eq];
        const value = arg[eq + 1 ..];
        if (std.mem.eql(u8, key, "--tokenizer")) tokenizer_path = value else if (std.mem.eql(u8, key, "--text")) text_path = value else if (std.mem.eql(u8, key, "--expected")) expected_path = value else if (std.mem.eql(u8, key, "--iterations")) iterations = try std.fmt.parseInt(usize, value, 10) else if (std.mem.eql(u8, key, "--warmups")) warmups = try std.fmt.parseInt(usize, value, 10) else return error.InvalidArgument;
    }
    if (iterations == 0) return error.InvalidArgument;
    const text = try std.Io.Dir.cwd().readFileAlloc(io, text_path orelse return error.MissingText, allocator, .unlimited);
    defer allocator.free(text);
    const expected = try std.Io.Dir.cwd().readFileAlloc(io, expected_path orelse return error.MissingReference, allocator, .unlimited);
    defer allocator.free(expected);
    if (@mod(expected.len, 4) != 0) return error.InvalidReference;
    var tokenizer = try tokenizer_lib.Tokenizer.fromFile(allocator, io, tokenizer_path orelse return error.MissingTokenizer);
    defer tokenizer.deinit();
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();
    const samples = try allocator.alloc(i96, iterations);
    defer allocator.free(samples);
    for (0..warmups + iterations) |iteration| {
        encoder.reset();
        const start = std.Io.Clock.awake.now(io);
        const ids = try encoder.encodeAlloc(allocator, text);
        const elapsed = start.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();
        defer allocator.free(ids);
        if (ids.len != expected.len / 4) return error.TokenCountMismatch;
        for (ids, 0..) |id, i| if (id != std.mem.readInt(u32, expected[i * 4 ..][0..4], .little)) return error.TokenIdMismatch;
        if (iteration >= warmups) samples[iteration - warmups] = elapsed;
    }
    const ordered = try allocator.dupe(i96, samples);
    defer allocator.free(ordered);
    std.mem.sort(i96, ordered, {}, std.sort.asc(i96));
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(expected, &hash, .{});
    const result = .{
        .bytes = text.len,
        .tokens = expected.len / 4,
        .warmups = warmups,
        .iterations = iterations,
        .median_ns = ordered[ordered.len / 2],
        .min_ns = ordered[0],
        .samples_ns = samples,
        .reference_sha256 = std.fmt.bytesToHex(hash, .lower),
        .every_iteration_matches_reference = true,
    };
    const output = try std.json.Stringify.valueAlloc(allocator, result, .{});
    defer allocator.free(output);
    std.debug.print("{s}\n", .{output});
}
