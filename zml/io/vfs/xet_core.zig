//! Native XET download support for ZML.
//!
//! HuggingFace serves large files through the XET content-addressed storage
//! (CAS) protocol. The default `hf://` path in `vfs/hf.zig` downloads the fully
//! reconstructed object through the `resolve/` compatibility bridge, which works
//! but does no client-side deduplication and re-fetches everything on every run.
//!
//! This module talks to the XET CAS directly using the `jedisct1/zig-xet`
//! implementation of the protocol (chunk reconstruction, LZ4/BG4 decompression,
//! BLAKE3 verification). It exposes:
//!   - `requestReadToken`: exchange an HF token for a CAS read token.
//!   - `Session`: a CAS client + file reconstructor, with `readRange` for
//!     positional reads (used by `vfs/xet.zig`) and `downloadToWriter` for
//!     streaming a whole file.
//!
//! Everything is driven by `std.Io`, matching ZML's async substrate, so the
//! parallel fetcher shares the same concurrency model as the rest of `zml/io`.

const std = @import("std");
const xet = @import("xet");

const cas_client = xet.cas_client;
const reconstruction = xet.reconstruction;

const log = std.log.scoped(.@"zml/io/xet");

/// A CAS read token obtained from the HuggingFace Hub. `access_token` and
/// `cas_url` are owned by this struct.
pub const ReadToken = struct {
    access_token: []const u8,
    cas_url: []const u8,
    /// Unix expiry timestamp for the access token.
    exp: i64,

    pub fn deinit(self: ReadToken, allocator: std.mem.Allocator) void {
        allocator.free(self.access_token);
        allocator.free(self.cas_url);
    }
};

/// Exchange an HF token for a XET CAS read token (accessToken + casUrl).
///
/// Hits `https://huggingface.co/api/{repo_type}s/{repo_id}/xet-read-token/{rev}`.
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

    var http_client = std.http.Client{ .allocator = allocator, .io = io };
    defer http_client.deinit();

    const auth = try std.fmt.allocPrint(allocator, "Bearer {s}", .{hf_token});
    defer allocator.free(auth);
    const extra_headers = [_]std.http.Header{.{ .name = "Authorization", .value = auth }};

    const uri = try std.Uri.parse(url);
    var req = try http_client.request(.GET, uri, .{ .extra_headers = &extra_headers });
    defer req.deinit();

    try req.sendBodiless();
    var response = try req.receiveHead(&.{});
    if (response.head.status != .ok) {
        log.err("xet-read-token request failed: {s}", .{@tagName(response.head.status)});
        return error.XetAuthFailed;
    }

    const body = try cas_client.readBodyDecompressing(&response, allocator, 16 * 1024);
    defer allocator.free(body);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();

    const root = parsed.value.object;
    const access_token = root.get("accessToken") orelse return error.XetTokenMalformed;
    const cas_url = root.get("casUrl") orelse return error.XetTokenMalformed;
    const exp = root.get("exp") orelse return error.XetTokenMalformed;

    return .{
        .access_token = try allocator.dupe(u8, access_token.string),
        .cas_url = try allocator.dupe(u8, cas_url.string),
        .exp = exp.integer,
    };
}

/// A live XET CAS session: an HTTP CAS client plus a file reconstructor.
///
/// Cheap to keep open for the lifetime of a VFS provider; a single session can
/// reconstruct ranges of any file whose XET hash is known to the same CAS.
pub const Session = struct {
    allocator: std.mem.Allocator,
    cas: cas_client.CasClient,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        cas_url: []const u8,
        access_token: []const u8,
    ) !Session {
        return .{
            .allocator = allocator,
            .cas = try cas_client.CasClient.init(allocator, io, cas_url, access_token),
        };
    }

    pub fn deinit(self: *Session) void {
        self.cas.deinit();
    }

    fn reconstructor(self: *Session) reconstruction.FileReconstructor {
        return reconstruction.FileReconstructor.init(self.allocator, &self.cas);
    }

    /// Reconstruct the byte range `[start, end)` of the file identified by its
    /// 32-byte XET hash. Only the xorbs overlapping the range are fetched.
    /// Caller owns the returned slice.
    pub fn readRange(self: *Session, file_hash: [32]u8, start: u64, end: u64) ![]u8 {
        var recon = self.reconstructor();
        return recon.reconstructRange(file_hash, start, end);
    }

    /// Reconstruct an entire file and stream it to `writer`, fetching xorbs in
    /// parallel. `verify` toggles BLAKE3 verification during reconstruction.
    pub fn downloadToWriter(
        self: *Session,
        file_hash: [32]u8,
        writer: *std.Io.Writer,
        verify: bool,
    ) !void {
        var recon = self.reconstructor();
        return recon.reconstructStreamParallel(file_hash, writer, verify, null);
    }
};

/// Parse a 64-char hex XET hash (as returned by the Hub tree API `xetHash`
/// field or the `x-xet-hash` header) into a 32-byte hash.
pub fn parseHash(hex: []const u8) ![32]u8 {
    return cas_client.apiHexToHash(hex);
}

/// One XET-backed file in a repository.
pub const FileEntry = struct {
    /// Full repository path.
    path: []const u8,
    size: u64,
    /// 64-char hex XET hash.
    hash_hex: []const u8,
};

/// List the XET-backed files of a repository. Non-XET files are skipped.
/// Caller owns the returned slice and each entry's `path`/`hash_hex`.
pub fn listXetFiles(
    allocator: std.mem.Allocator,
    io: std.Io,
    environ: std.process.Environ,
    repo_type: []const u8,
    repo_id: []const u8,
    revision: []const u8,
    hf_token: []const u8,
) ![]FileEntry {
    var file_list = try xet.model_download.listFiles(
        allocator,
        io,
        environ,
        repo_id,
        repo_type,
        revision,
        hf_token,
    );
    defer file_list.deinit();

    var entries: std.ArrayList(FileEntry) = .empty;
    errdefer {
        for (entries.items) |e| {
            allocator.free(e.path);
            allocator.free(e.hash_hex);
        }
        entries.deinit(allocator);
    }

    for (file_list.files) |file| {
        const hash = file.xet_hash orelse continue;
        try entries.append(allocator, .{
            .path = try allocator.dupe(u8, file.path),
            .size = file.size,
            .hash_hex = try allocator.dupe(u8, hash),
        });
    }

    return entries.toOwnedSlice(allocator);
}
