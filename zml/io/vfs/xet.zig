//! Native XET download for ZML, backed by xet-core (Rust) through its C-API.
//!
//! Rather than reimplementing the XET protocol in Zig, this drives the mature
//! xet-core client via `xet_capi.zig` (the `hf_xet.h` C-API). You get the same
//! throughput as the official `hf-xet` client (adaptive concurrency, streaming,
//! decompress-once, on-disk chunk cache, BLAKE3 verification) at line rate.
//!
//! The C-API is whole-file oriented (`download_to_path`), so the integration
//! model is download-to-cache: materialize a repo file locally, then open it
//! through the normal filesystem. This mirrors how `huggingface_hub` + `hf-xet`
//! work. Set `HF_XET_HIGH_PERFORMANCE=1` for peak concurrency/buffers.
//!
//! Pieces:
//!   - `xet_hub.zig`  : HF Hub HTTPS/JSON (token exchange, file listing).
//!   - `xet_capi.zig` : Zig binding over xet-core's C-API.

const std = @import("std");

pub const hub = @import("xet_hub.zig");
pub const capi = @import("xet_capi.zig");

pub const ReadToken = hub.ReadToken;
pub const FileEntry = hub.FileEntry;
pub const Session = capi.Session;
pub const requestReadToken = hub.requestReadToken;
pub const listXetFiles = hub.listXetFiles;

const log = std.log.scoped(.@"zml/io/vfs/xet");

/// Download a single XET-backed repo file to `dest_path` via xet-core.
///
/// Resolves the file's XET hash + size from the Hub tree, exchanges the HF
/// token for a CAS read token, then reconstructs the file through the C-API.
pub fn downloadFile(
    allocator: std.mem.Allocator,
    io: std.Io,
    repo_type: []const u8,
    repo_id: []const u8,
    revision: []const u8,
    filepath: []const u8,
    hf_token: []const u8,
    dest_path: [:0]const u8,
) !void {
    const files = try hub.listXetFiles(allocator, io, repo_type, repo_id, revision, hf_token);
    defer {
        for (files) |f| {
            allocator.free(f.path);
            allocator.free(f.hash_hexz);
        }
        allocator.free(files);
    }

    var chosen: ?FileEntry = null;
    for (files) |f| {
        if (std.mem.eql(u8, f.path, filepath) or std.mem.endsWith(u8, f.path, filepath)) {
            chosen = f;
            break;
        }
    }
    const file = chosen orelse {
        log.err("{s} is not a XET-backed file in {s}", .{ filepath, repo_id });
        return error.FileNotXetBacked;
    };

    const token = try hub.requestReadToken(allocator, io, repo_type, repo_id, revision, hf_token);
    defer token.deinit(allocator);

    const cas_z = try allocator.dupeZ(u8, token.cas_url);
    defer allocator.free(cas_z);
    const tok_z = try allocator.dupeZ(u8, token.access_token);
    defer allocator.free(tok_z);

    var session = try Session.init(cas_z, tok_z, token.exp);
    defer session.deinit();

    try session.downloadToPath(file.hash_hexz, file.size, dest_path);
}
