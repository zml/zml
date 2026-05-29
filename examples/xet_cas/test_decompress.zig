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
    const xorb_path: []const u8 = it.next() orelse {
        std.debug.print("Usage: test_decompress <path-to-xorb-file>\n", .{});
        std.process.exit(1);
    };

    // Read xorb binary.
    const xorb_file = try std.Io.Dir.openFileAbsolute(io_ctx, xorb_path, .{ .mode = .read_only });
    defer xorb_file.close(io_ctx);
    const xorb_len = try xorb_file.length(io_ctx);
    var xorb_reader = xorb_file.reader(io_ctx, &.{});
    const xorb_data = try xorb_reader.interface.readAlloc(allocator, xorb_len);
    defer allocator.free(xorb_data);
    std.debug.print("Read {d} bytes from xorb file\n", .{xorb_data.len});

    // Allocate reusable decompression buffers (max chunk = 128 KiB).
    const max_chunk_size = 128 * 1024;
    const dst = try allocator.alloc(u8, max_chunk_size);
    defer allocator.free(dst);
    const tmp = try allocator.alloc(u8, max_chunk_size);
    defer allocator.free(tmp);

    // Iterate and decompress all chunks.
    var chunk_iter = io.xet.ChunkIterator{ .data = xorb_data };
    var chunk_count: u32 = 0;
    var total_decompressed: u64 = 0;
    var type_histogram = [_]u32{ 0, 0, 0, 0 };
    var errors: u32 = 0;

    while (try chunk_iter.next()) |chunk| {
        const result = io.xet.decompressChunk(chunk, dst, tmp) catch |err| {
            std.debug.print("ERROR: chunk {d} (type {d}): {s}\n", .{ chunk.index, chunk.compression_type, @errorName(err) });
            errors += 1;
            chunk_count += 1;
            continue;
        };

        // Verify output length matches header.
        if (result.len != chunk.uncompressed_size) {
            std.debug.print("ERROR: chunk {d}: decompressed {d} bytes, expected {d}\n", .{ chunk.index, result.len, chunk.uncompressed_size });
            errors += 1;
        }

        total_decompressed += result.len;
        type_histogram[chunk.compression_type] += 1;
        chunk_count += 1;
    }

    // Summary.
    std.debug.print("\n--- Summary ---\n", .{});
    std.debug.print("Chunks processed:      {d}\n", .{chunk_count});
    std.debug.print("Total decompressed:    {d} bytes\n", .{total_decompressed});
    std.debug.print("Errors:                {d}\n", .{errors});

    const type_names = [_][]const u8{ "None", "LZ4", "ByteGrouping4LZ4", "FullBitsliceLZ4" };
    std.debug.print("\nCompression types:\n", .{});
    for (type_histogram, 0..) |count, i| {
        if (count > 0) {
            std.debug.print("  {s}: {d}\n", .{ type_names[i], count });
        }
    }

    if (errors > 0) {
        std.debug.print("\nFAIL: {d} chunks failed decompression\n", .{errors});
        std.process.exit(1);
    }
    std.debug.print("\nPASS: all {d} chunks decompressed successfully\n", .{chunk_count});
}
