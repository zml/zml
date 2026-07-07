//! HuggingFace Hub helpers for XET downloads (pure Zig, std only).
//!
//! These resolve the two things the XET CAS client needs that live on the Hub
//! side rather than in the CAS protocol itself:
//!   - `requestReadToken`: exchange an HF token for a CAS read token
//!     (accessToken + casUrl + expiry).
//!   - `listXetFiles`: enumerate a repo's XET-backed files with their hash+size.
//!
//! The actual chunk reconstruction is done by xet-core through the C-API (see
//! `xet_capi.zig`); this module only does plain HTTPS + JSON. Both calls share
//! a caller-owned `std.http.Client` (connection reuse), like the other VFS
//! backends in this directory.

const std = @import("std");

const log = std.log.scoped(.@"zml/io/vfs/xet_hub");

/// A bearer-token authorization value, e.g. `.{ .override = "Bearer <token>" }`.
pub const Auth = std.http.Client.Request.Headers.Value;

fn readBody(response: *std.http.Client.Response, allocator: std.mem.Allocator, max: usize) ![]u8 {
    var transfer_buffer: [16 * 1024]u8 = undefined;
    var decompress_buffer: [std.compress.flate.max_window_len]u8 = undefined;
    var decompress: std.http.Decompress = undefined;
    var reader = response.readerDecompressing(&transfer_buffer, &decompress, &decompress_buffer);
    return reader.allocRemaining(allocator, std.Io.Limit.limited(max));
}

/// GET `url` with bearer auth and parse the response body as JSON. Caller owns
/// the returned parse tree (`defer parsed.deinit()`).
fn getJson(
    allocator: std.mem.Allocator,
    client: *std.http.Client,
    url: []const u8,
    auth: Auth,
    max_body: usize,
) !std.json.Parsed(std.json.Value) {
    const uri = try std.Uri.parse(url);
    var req = try client.request(.GET, uri, .{ .headers = .{ .authorization = auth } });
    defer req.deinit();
    try req.sendBodiless();

    var redirect_buffer: [8 * 1024]u8 = undefined;
    var response = try req.receiveHead(&redirect_buffer);
    if (response.head.status != .ok) {
        log.err("{s} -> {s}", .{ url, @tagName(response.head.status) });
        return error.XetHubRequestFailed;
    }

    const body = try readBody(&response, allocator, max_body);
    defer allocator.free(body);
    return std.json.parseFromSlice(std.json.Value, allocator, body, .{});
}

pub const ReadToken = struct {
    access_token: [:0]const u8,
    cas_url: [:0]const u8,
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
    client: *std.http.Client,
    repo_type: []const u8,
    repo_id: []const u8,
    revision: []const u8,
    auth: Auth,
) !ReadToken {
    const url = try std.fmt.allocPrint(
        allocator,
        "https://huggingface.co/api/{s}s/{s}/xet-read-token/{s}",
        .{ repo_type, repo_id, revision },
    );
    defer allocator.free(url);

    const parsed = try getJson(allocator, client, url, auth, 16 * 1024);
    defer parsed.deinit();

    const root = parsed.value.object;
    const access = root.get("accessToken") orelse return error.XetTokenMalformed;
    const cas = root.get("casUrl") orelse return error.XetTokenMalformed;
    const exp = root.get("exp") orelse return error.XetTokenMalformed;
    if (access != .string or cas != .string or exp != .integer) return error.XetTokenMalformed;

    const access_token = try allocator.dupeZ(u8, access.string);
    errdefer allocator.free(access_token);
    const cas_url = try allocator.dupeZ(u8, cas.string);
    return .{ .access_token = access_token, .cas_url = cas_url, .exp = exp.integer };
}

pub const XetFile = struct {
    size: u64,
    /// 64-char hex XET hash, NUL-terminated for direct C-API use. Caller owns it.
    hash_hexz: [:0]const u8,
};

/// Resolve `filepath` to its XET hash + size from the repo tree. Returns
/// `error.FileNotXetBacked` if the file is absent or not XET-backed. Caller
/// owns the returned `hash_hexz`.
pub fn findXetFile(
    allocator: std.mem.Allocator,
    client: *std.http.Client,
    repo_type: []const u8,
    repo_id: []const u8,
    revision: []const u8,
    filepath: []const u8,
    auth: Auth,
) !XetFile {
    const url = try std.fmt.allocPrint(
        allocator,
        "https://huggingface.co/api/{s}s/{s}/tree/{s}?recursive=true",
        .{ repo_type, repo_id, revision },
    );
    defer allocator.free(url);

    const parsed = try getJson(allocator, client, url, auth, 8 * 1024 * 1024);
    defer parsed.deinit();

    for (parsed.value.array.items) |item| {
        const obj = item.object;
        const path = obj.get("path") orelse continue;
        const xet = obj.get("xetHash") orelse continue;
        const size = obj.get("size") orelse continue;
        if (path != .string or xet != .string or size != .integer) continue;
        if (!std.mem.eql(u8, path.string, filepath)) continue;
        return .{ .size = @intCast(size.integer), .hash_hexz = try allocator.dupeZ(u8, xet.string) };
    }
    return error.FileNotXetBacked;
}
