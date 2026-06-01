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
    const usage = "Usage: test_reconstruct <group.json> <xorb-file> <xorb-hash> <start-chunk-index>\n";
    const json_path: []const u8 = it.next() orelse {
        std.debug.print(usage, .{});
        std.process.exit(1);
    };
    const xorb_path: []const u8 = it.next() orelse {
        std.debug.print(usage, .{});
        std.process.exit(1);
    };
    const xorb_hash: []const u8 = it.next() orelse {
        std.debug.print(usage, .{});
        std.process.exit(1);
    };
    const start_chunk_index: u32 = blk: {
        const s = it.next() orelse {
            std.debug.print(usage, .{});
            std.process.exit(1);
        };
        break :blk std.fmt.parseInt(u32, s, 10) catch {
            std.debug.print("Invalid start-chunk-index: {s}\n", .{s});
            std.process.exit(1);
        };
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
    std.debug.print("Parsed {d} terms\n", .{terms.len});

    // Read xorb binary.
    const xorb_file = try std.Io.Dir.openFileAbsolute(io_ctx, xorb_path, .{ .mode = .read_only });
    defer xorb_file.close(io_ctx);
    var xorb_reader = xorb_file.reader(io_ctx, &.{});
    const xorb_data = try xorb_reader.interface.readAlloc(allocator, try xorb_file.length(io_ctx));
    defer allocator.free(xorb_data);
    std.debug.print("Read {d} bytes from xorb file\n", .{xorb_data.len});

    // Build xorb map.
    var arena_impl = std.heap.ArenaAllocator.init(allocator);
    defer arena_impl.deinit();
    const arena = arena_impl.allocator();
    const map = try io.xet.buildXorbMap(arena, terms);
    std.debug.print("{d} unique xorb entries\n", .{map.entries.len});

    // Find the xorb entry matching the provided hash.
    const entry = blk: {
        for (map.entries) |e| {
            if (std.mem.eql(u8, e.hash, xorb_hash)) break :blk e;
        }
        std.debug.print("FAIL: no xorb entry for hash {s}\n", .{xorb_hash});
        std.process.exit(1);
    };
    std.debug.print("Using xorb entry with hash prefix {s}... ({d} mapped chunks)\n", .{
        entry.hash[0..@min(16, entry.hash.len)],
        entry.chunk_to_terms.len,
    });

    // Allocate per-term byte counters.
    const counters = try allocator.alloc(u64, terms.len);
    defer allocator.free(counters);
    @memset(counters, 0);

    // Allocate scratch buffers.
    const max_chunk_size = 128 * 1024;
    const dst = try allocator.alloc(u8, max_chunk_size);
    defer allocator.free(dst);
    const tmp = try allocator.alloc(u8, max_chunk_size);
    defer allocator.free(tmp);

    // Process xorb.
    const Ctx = struct {
        counters: []u64,

        fn onChunk(self: *@This(), term_idx: u32, data: []const u8) !void {
            self.counters[term_idx] += data.len;
        }
    };
    var ctx = Ctx{ .counters = counters };
    try io.xet.processXorb(xorb_data, entry, start_chunk_index, dst, tmp, &ctx, Ctx.onChunk);

    // Verify: each referenced term's counter == unpacked_length.
    var terms_validated: u32 = 0;
    var total_bytes: u64 = 0;
    var failures: u32 = 0;

    for (terms, 0..) |term, i| {
        if (counters[i] == 0 and !std.mem.eql(u8, term.hash, entry.hash)) continue;
        if (counters[i] == 0) continue;

        terms_validated += 1;
        total_bytes += counters[i];

        if (counters[i] != term.unpacked_length) {
            std.debug.print("FAIL: term {d} (hash {s}): got {d} bytes, expected {d}\n", .{
                i, term.hash[0..@min(16, term.hash.len)], counters[i], term.unpacked_length,
            });
            failures += 1;
        }
    }

    // Summary.
    std.debug.print("\n--- Summary ---\n", .{});
    std.debug.print("Terms validated:  {d}\n", .{terms_validated});
    std.debug.print("Total bytes:      {d}\n", .{total_bytes});
    std.debug.print("Failures:         {d}\n", .{failures});

    if (failures > 0) {
        std.debug.print("\nFAIL: {d} terms had incorrect byte counts\n", .{failures});
        std.process.exit(1);
    }
    std.debug.print("\nPASS: all {d} validated terms match expected unpacked_length\n", .{terms_validated});
}
