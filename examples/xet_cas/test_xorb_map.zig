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
        std.debug.print("Usage: test_xorb_map <path-to-group_000.json>\n", .{});
        std.process.exit(1);
    };

    const file = try std.Io.Dir.openFileAbsolute(io_ctx, json_path, .{ .mode = .read_only });
    defer file.close(io_ctx);
    var reader = file.reader(io_ctx, &.{});
    const json_bytes = try reader.interface.readAlloc(allocator, try file.length(io_ctx));
    defer allocator.free(json_bytes);

    const parsed = try std.json.parseFromSlice(io.xet.ReconstructionResponse, allocator, json_bytes, .{
        .ignore_unknown_fields = true,
    });
    defer parsed.deinit();

    const terms = parsed.value.terms;
    std.debug.print("Parsed {d} terms\n", .{terms.len});

    var arena_impl = std.heap.ArenaAllocator.init(allocator);
    defer arena_impl.deinit();
    const arena = arena_impl.allocator();

    const map = try io.xet.buildXorbMap(arena, terms);
    std.debug.print("{d} unique xorb entries\n", .{map.entries.len});

    // Invariant 1: every term's chunk range is fully covered.
    for (terms, 0..) |term, term_idx| {
        const ti: u32 = @intCast(term_idx);
        const entry = findEntry(map, term.hash) orelse {
            std.debug.print("FAIL: no xorb entry for term {d} hash {s}\n", .{ term_idx, term.hash });
            std.process.exit(1);
        };
        for (@as(usize, @intCast(term.range.start))..@as(usize, @intCast(term.range.end))) |c| {
            var found = false;
            for (entry.chunk_to_terms[c]) |ref_ti| {
                if (ref_ti == ti) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                std.debug.print("FAIL: term {d} chunk {d} not found in xorb map\n", .{ term_idx, c });
                std.process.exit(1);
            }
        }
    }

    // Invariant 2: total chunk refs == sum(end - start).
    var total_refs: usize = 0;
    for (map.entries) |entry| {
        for (entry.chunk_to_terms) |term_list| {
            total_refs += term_list.len;
        }
    }
    var expected_refs: usize = 0;
    for (terms) |t| expected_refs += @intCast(t.range.end - t.range.start);
    if (total_refs != expected_refs) {
        std.debug.print("FAIL: total refs {d} != expected {d}\n", .{ total_refs, expected_refs });
        std.process.exit(1);
    }

    // Invariant 3: no term index appears in a chunk outside its range.
    for (map.entries) |entry| {
        for (entry.chunk_to_terms, 0..) |term_list, chunk_idx| {
            for (term_list) |ti| {
                const t = terms[ti];
                if (chunk_idx < t.range.start or chunk_idx >= t.range.end) {
                    std.debug.print("FAIL: term {d} at chunk {d} outside range [{d},{d})\n", .{ ti, chunk_idx, t.range.start, t.range.end });
                    std.process.exit(1);
                }
            }
        }
    }

    std.debug.print("All invariants passed: {d} terms, {d} xorbs, {d} total refs\n", .{ terms.len, map.entries.len, total_refs });
}

fn findEntry(map: io.xet.XorbMap, hash: []const u8) ?io.xet.XorbMap.Entry {
    for (map.entries) |entry| {
        if (std.mem.eql(u8, entry.hash, hash)) return entry;
    }
    return null;
}
