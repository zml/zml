//! HuggingFace Hub helpers for XET downloads (pure Zig, std only).
//!
//! These resolve the two things the XET CAS client needs that live on the Hub
//! side rather than in the CAS protocol itself:
//!   - `requestReadToken`: exchange an HF token for a CAS read token
//!     (accessToken + casUrl + expiry).
//!   - `listXetFiles`: enumerate a repo's XET-backed files with their hash+size.
//!
//! The actual chunk reconstruction is done by xet-core through the C-API (see
//! `xet_capi.zig`); this module only does plain HTTPS + JSON.

const std = @import("std");

const log = std.log.scoped(.@"zml/io/vfs/xet_hub");

fn readBody(response: *std.http.Client.Response, allocator: std.mem.Allocator, max: usize) ![]u8 {
    var transfer_buffer: [16 * 1024]u8 = undefined;
    var decompress_buffer: [std.compress.flate.max_window_len]u8 = undefined;
    var decompress: std.http.Decompress = undefined;
    var reader = response.readerDecompressing(&transfer_buffer, &decompress, &decompress_buffer);
    return reader.allocRemaining(allocator, std.Io.Limit.limited(max));
}

pub const ReadToken = struct {
    access_token: []const u8,
    cas_url: []const u8,
    exp: i64,

    pub fn deinit(self: ReadToken, allocator: std.mem.Allocator) void {
        allocator.free(self.access_token);
        allocator.free(self.cas_url);
    }
};

/// Exchange an HF token for a CAS read token via
/// `https://huggingface.co/api/{repo_type}s/{repo_id}/xet-read-token/{rev}`.
pub fn requestReadToken(
    allocator: std.mem.Allocator,
    io: std.Io,
    repo_type: []const u8,
    repo_id: []const u8,
    revision: []const u8,
    hf_token: []const u8,
) !ReadToken {
    const url = try std.fmt.allocPrint(
        allocator,
        "https://huggingface.co/api/{s}s/{s}/xet-read-token/{s}",
        .{ repo_type, repo_id, revision },
    );
    defer allocator.free(url);

    var client = std.http.Client{ .allocator = allocator, .io = io };
    defer client.deinit();

    const auth = try std.fmt.allocPrint(allocator, "Bearer {s}", .{hf_token});
    defer allocator.free(auth);
    const extra = [_]std.http.Header{.{ .name = "Authorization", .value = auth }};

    const uri = try std.Uri.parse(url);
    var req = try client.request(.GET, uri, .{ .extra_headers = &extra });
    defer req.deinit();
    try req.sendBodiless();
    var response = try req.receiveHead(&.{});
    if (response.head.status != .ok) {
        log.err("xet-read-token failed: {s}", .{@tagName(response.head.status)});
        return error.XetAuthFailed;
    }

    const body = try readBody(&response, allocator, 16 * 1024);
    defer allocator.free(body);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();
    const root = parsed.value.object;

    return .{
        .access_token = try allocator.dupe(u8, (root.get("accessToken") orelse return error.XetTokenMalformed).string),
        .cas_url = try allocator.dupe(u8, (root.get("casUrl") orelse return error.XetTokenMalformed).string),
        .exp = (root.get("exp") orelse return error.XetTokenMalformed).integer,
    };
}

pub const FileEntry = struct {
    path: []const u8,
    size: u64,
    /// 64-char hex XET hash, NUL-terminated for direct C-API use.
    hash_hexz: [:0]const u8,
};

/// List the XET-backed files of a repo (non-XET files are skipped).
/// Caller owns the returned slice and each entry's `path` / `hash_hexz`.
pub fn listXetFiles(
    allocator: std.mem.Allocator,
    io: std.Io,
    repo_type: []const u8,
    repo_id: []const u8,
    revision: []const u8,
    hf_token: []const u8,
) ![]FileEntry {
    const url = try std.fmt.allocPrint(
        allocator,
        "https://huggingface.co/api/{s}s/{s}/tree/{s}?recursive=true",
        .{ repo_type, repo_id, revision },
    );
    defer allocator.free(url);

    var client = std.http.Client{ .allocator = allocator, .io = io };
    defer client.deinit();

    const auth = try std.fmt.allocPrint(allocator, "Bearer {s}", .{hf_token});
    defer allocator.free(auth);
    const extra = [_]std.http.Header{.{ .name = "Authorization", .value = auth }};

    const uri = try std.Uri.parse(url);
    var req = try client.request(.GET, uri, .{ .extra_headers = &extra });
    defer req.deinit();
    try req.sendBodiless();
    var response = try req.receiveHead(&.{});
    if (response.head.status != .ok) return error.TreeRequestFailed;

    const body = try readBody(&response, allocator, 8 * 1024 * 1024);
    defer allocator.free(body);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();

    var entries: std.ArrayList(FileEntry) = .empty;
    errdefer {
        for (entries.items) |e| {
            allocator.free(e.path);
            allocator.free(e.hash_hexz);
        }
        entries.deinit(allocator);
    }

    for (parsed.value.array.items) |item| {
        const obj = item.object;
        const xet = obj.get("xetHash") orelse continue;
        if (xet != .string) continue;
        try entries.append(allocator, .{
            .path = try allocator.dupe(u8, (obj.get("path") orelse continue).string),
            .size = @intCast((obj.get("size") orelse continue).integer),
            .hash_hexz = try allocator.dupeZ(u8, xet.string),
        });
    }

    return entries.toOwnedSlice(allocator);
}
