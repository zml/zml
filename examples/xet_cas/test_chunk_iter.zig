const std = @import("std");
const io = @import("io");

pub const std_options: std.Options = .{
    .log_level = .info,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io_ctx = init.io;

    var it = init.minimal.args.iterate();
    _ = it.skip(); // program name
    const json_path: []const u8 = it.next() orelse {
        std.debug.print("Usage: test_chunk_iter <path-to-group_000.json> <path-to-xorb-file>\n", .{});
        std.process.exit(1);
    };
    const xorb_path: []const u8 = it.next() orelse {
        std.debug.print("Usage: test_chunk_iter <path-to-group_000.json> <path-to-xorb-file>\n", .{});
        std.process.exit(1);
    };

    // Parse reconstruction JSON.
    const json_file = try std.Io.Dir.openFileAbsolute(io_ctx, json_path, .{ .mode = .read_only });
    defer json_file.close(io_ctx);
    var json_reader = json_file.reader(io_ctx, &.{});
    const json_bytes = try json_reader.interface.readAlloc(allocator, try json_file.length(io_ctx));
    defer allocator.free(json_bytes);

    const parsed = try std.json.parseFromSlice(io.xet.ReconstructionResponse, allocator, json_bytes, .{
        .ignore_unknown_fields = true,
    });
    defer parsed.deinit();
    const terms = parsed.value.terms;
    std.debug.print("Parsed {d} terms from JSON\n", .{terms.len});

    // Read xorb binary.
    const xorb_file = try std.Io.Dir.openFileAbsolute(io_ctx, xorb_path, .{ .mode = .read_only });
    defer xorb_file.close(io_ctx);
    const xorb_len = try xorb_file.length(io_ctx);
    var xorb_reader = xorb_file.reader(io_ctx, &.{});
    const xorb_data = try xorb_reader.interface.readAlloc(allocator, xorb_len);
    defer allocator.free(xorb_data);
    std.debug.print("Read {d} bytes from xorb file\n", .{xorb_data.len});

    // Iterate all chunks.
    var chunk_iter = io.xet.ChunkIterator{ .data = xorb_data };
    var chunk_count: u32 = 0;
    var total_compressed: u64 = 0;
    var total_uncompressed: u64 = 0;
    var total_wire: u64 = 0;
    var type_histogram = [_]u32{ 0, 0, 0, 0 };

    while (try chunk_iter.next()) |chunk| {
        total_compressed += chunk.compressed_size;
        total_uncompressed += chunk.uncompressed_size;
        total_wire += io.xet.ChunkIterator.header_size + chunk.compressed_size;
        type_histogram[chunk.compression_type] += 1;
        chunk_count += 1;
    }

    std.debug.print("\n--- Summary ---\n", .{});
    std.debug.print("Chunks parsed:        {d}\n", .{chunk_count});
    std.debug.print("Total compressed:     {d} bytes\n", .{total_compressed});
    std.debug.print("Total uncompressed:   {d} bytes\n", .{total_uncompressed});
    std.debug.print("Total wire (hdr+data):{d} bytes\n", .{total_wire});
    std.debug.print("File size:            {d} bytes\n", .{xorb_data.len});

    // Verify: wire size == file size.
    if (total_wire != xorb_data.len) {
        std.debug.print("FAIL: wire size {d} != file size {d}\n", .{ total_wire, xorb_data.len });
        std.process.exit(1);
    }
    std.debug.print("PASS: wire size matches file size\n", .{});

    // Compression type histogram.
    const type_names = [_][]const u8{ "None", "LZ4", "ByteGrouping4LZ4", "FullBitsliceLZ4" };
    std.debug.print("\nCompression types:\n", .{});
    for (type_histogram, 0..) |count, i| {
        if (count > 0) {
            std.debug.print("  {s}: {d}\n", .{ type_names[i], count });
        }
    }

    // Try to match chunk count against a fetch_info entry covering this xorb.
    // Use the first term's hash to find matching terms, then check if any
    // term's range covers exactly [0, chunk_count).
    if (terms.len > 0) {
        var matched = false;
        for (terms) |term| {
            if (term.range.start == 0 and term.range.end == chunk_count) {
                std.debug.print("\nChunk count {d} matches term with hash {s} range [0,{d})\n", .{ chunk_count, term.hash, term.range.end });
                matched = true;
                break;
            }
        }
        if (!matched) {
            std.debug.print("\nNote: no single term has range [0,{d}) — xorb may span multiple terms or a subset was downloaded\n", .{chunk_count});
        }
    }
}
