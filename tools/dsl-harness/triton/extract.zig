//! Slice TTIR/TTGIR out of XLA's per-pass dump file. Each snapshot starts
//! with `// -----// IR Dump Before <PassName>`; we take the last one
//! before `ConvertTritonToTritonGPU` (TTIR) and the last before
//! `ConvertTritonGPUToLLVM` (TTGIR). LLIR/PTX live in adjacent
//! `module_NNNN.<symbol>.0.{ir-with-opt.ll,ptx}` files.

const std = @import("std");

const Allocator = std.mem.Allocator;

const PASS_TTIR = "ConvertTritonToTritonGPU";
const PASS_TTGIR = "ConvertTritonGPUToLLVM";

pub const Snapshot = struct {
    pass_name: []const u8,
    /// Slice into the original `text` — no copy.
    body: []const u8,
};

fn passNameFromHeader(line: []const u8) ?[]const u8 {
    const prefix = "// -----// IR Dump Before ";
    if (!std.mem.startsWith(u8, line, prefix)) return null;
    const rest = line[prefix.len..];
    var i: usize = 0;
    while (i < rest.len) : (i += 1) {
        const c = rest[i];
        if (!std.ascii.isAlphanumeric(c) and c != '_') break;
    }
    if (i == 0) return null;
    return rest[0..i];
}

pub fn splitSnapshots(allocator: Allocator, text: []const u8) ![]Snapshot {
    var out: std.ArrayList(Snapshot) = .empty;
    errdefer out.deinit(allocator);

    var current_pass: ?[]const u8 = null;
    var body_start: usize = 0;
    var pos: usize = 0;

    while (pos < text.len) {
        const line_start = pos;
        const eol = std.mem.indexOfScalarPos(u8, text, pos, '\n') orelse text.len;
        const line = text[line_start..eol];

        if (passNameFromHeader(line)) |pass| {
            if (current_pass) |prev_pass| {
                try out.append(allocator, .{
                    .pass_name = prev_pass,
                    .body = text[body_start..line_start],
                });
            }
            current_pass = pass;
            body_start = if (eol < text.len) eol + 1 else eol;
        }

        pos = if (eol < text.len) eol + 1 else text.len;
    }

    if (current_pass) |prev_pass| {
        try out.append(allocator, .{
            .pass_name = prev_pass,
            .body = text[body_start..text.len],
        });
    }

    return out.toOwnedSlice(allocator);
}

fn lastBefore(snapshots: []const Snapshot, pass_name: []const u8) ?[]const u8 {
    var i: usize = snapshots.len;
    while (i > 0) {
        i -= 1;
        if (std.mem.eql(u8, snapshots[i].pass_name, pass_name)) {
            return snapshots[i].body;
        }
    }
    return null;
}

pub fn extractTtir(snapshots: []const Snapshot) ?[]const u8 {
    return lastBefore(snapshots, PASS_TTIR);
}

pub fn extractTtgir(snapshots: []const Snapshot) ?[]const u8 {
    return lastBefore(snapshots, PASS_TTGIR);
}

const LL_SUFFIX = ".0.ir-with-opt.ll";
const PTX_SUFFIX = ".0.ptx";

/// `module_NNNN.<kernel>.0.ir-with-opt.ll` → `<kernel>`.
pub fn matchLlirFilename(filename: []const u8) ?[]const u8 {
    return matchModuleFilename(filename, LL_SUFFIX);
}

/// `module_NNNN.<kernel>.0.ptx` → `<kernel>`.
pub fn matchPtxFilename(filename: []const u8) ?[]const u8 {
    return matchModuleFilename(filename, PTX_SUFFIX);
}

fn matchModuleFilename(filename: []const u8, suffix: []const u8) ?[]const u8 {
    if (!std.mem.startsWith(u8, filename, "module_")) return null;
    if (!std.mem.endsWith(u8, filename, suffix)) return null;
    var i: usize = "module_".len;
    while (i < filename.len and std.ascii.isDigit(filename[i])) i += 1;
    if (i >= filename.len or filename[i] != '.') return null;
    const kernel_start = i + 1;
    const kernel_end = filename.len - suffix.len;
    if (kernel_end <= kernel_start) return null;
    return filename[kernel_start..kernel_end];
}

test "passNameFromHeader extracts pass name" {
    try std.testing.expectEqualStrings("ConvertTritonToTritonGPU", passNameFromHeader("// -----// IR Dump Before ConvertTritonToTritonGPU (foo) //----- //").?);
    try std.testing.expectEqualStrings("Foo123_Bar", passNameFromHeader("// -----// IR Dump Before Foo123_Bar").?);
    try std.testing.expect(passNameFromHeader("// some other comment") == null);
    try std.testing.expect(passNameFromHeader("module {") == null);
}

test "splitSnapshots and lastBefore" {
    const allocator = std.testing.allocator;
    const text =
        "// preamble (skipped)\n" ++
        "// -----// IR Dump Before PassA (...) //----- //\n" ++
        "module { /* A1 */ }\n" ++
        "// -----// IR Dump Before PassB //----- //\n" ++
        "module { /* B */ }\n" ++
        "// -----// IR Dump Before PassA //----- //\n" ++
        "module { /* A2 */ }\n";
    const snaps = try splitSnapshots(allocator, text);
    defer allocator.free(snaps);
    try std.testing.expectEqual(@as(usize, 3), snaps.len);
    try std.testing.expectEqualStrings("PassA", snaps[0].pass_name);
    try std.testing.expect(std.mem.indexOf(u8, snaps[0].body, "A1").? > 0);
    try std.testing.expectEqualStrings("PassA", snaps[2].pass_name);

    const a_body = lastBefore(snaps, "PassA").?;
    try std.testing.expect(std.mem.indexOf(u8, a_body, "A2").? > 0);
    try std.testing.expect(lastBefore(snaps, "PassZ") == null);
}

test "matchLlirFilename / matchPtxFilename" {
    try std.testing.expectEqualStrings(
        "triton_add_kernel",
        matchLlirFilename("module_0042.triton_add_kernel.0.ir-with-opt.ll").?,
    );
    try std.testing.expectEqualStrings(
        "kernel_unified_attention_2d_ptr",
        matchPtxFilename("module_0001.kernel_unified_attention_2d_ptr.0.ptx").?,
    );
    try std.testing.expect(matchLlirFilename("foo.ll") == null);
    try std.testing.expect(matchPtxFilename("module_42.kernel.1.ptx") == null);
}
