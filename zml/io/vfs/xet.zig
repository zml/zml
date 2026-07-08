//! Native XET reads for ZML, backed by xet-core (Rust) through its C-API.
//!
//! Rather than reimplementing the XET protocol in Zig, this drives the mature
//! xet-core client via `xet_capi.zig`. You get the same throughput as the
//! official `hf-xet` client (adaptive concurrency, streaming, decompress-once,
//! on-disk chunk cache).
//!
//! `openRemote` resolves a repo file to a reusable `RemoteFile` whose
//! `readRange` reconstructs arbitrary byte ranges on demand. This is what
//! `hf.zig` uses to serve positional reads of `hf://` XET files.
//!
//! Set `HF_XET_HIGH_PERFORMANCE=1` for peak concurrency/buffers.

const std = @import("std");

const hub = @import("xet_hub.zig");
const capi = @import("xet_capi.zig");

/// A repo file resolved to a live xet-core session, ready for range reads.
pub const RemoteFile = struct {
    session: capi.Session,
    hash: [:0]const u8,
    size: u64,

    pub fn deinit(self: *RemoteFile, allocator: std.mem.Allocator) void {
        self.session.deinit();
        allocator.free(self.hash);
    }

    /// Reconstruct `[start, end)` into `dest`, returning bytes written.
    pub fn readRange(self: *RemoteFile, start: u64, end: u64, dest: []u8) !usize {
        return self.session.readRange(self.hash, self.size, start, end, dest);
    }
};

/// Repo-tree and read-token helpers, re-exported so callers (e.g. `hf.zig`) can
/// fetch each once per repo and cache them instead of per file.
pub const XetTree = hub.XetTree;
pub const XetFile = hub.XetFile;
pub const ReadToken = hub.ReadToken;
pub const Auth = hub.Auth;

pub fn fetchTree(
    allocator: std.mem.Allocator,
    client: *std.http.Client,
    auth: hub.Auth,
    repo_type: []const u8,
    repo_id: []const u8,
    revision: []const u8,
) !XetTree {
    return hub.fetchXetTree(allocator, client, repo_type, repo_id, revision, auth);
}

pub fn freeTree(allocator: std.mem.Allocator, tree: *XetTree) void {
    hub.freeXetTree(allocator, tree);
}

pub fn fetchToken(
    allocator: std.mem.Allocator,
    client: *std.http.Client,
    auth: hub.Auth,
    repo_type: []const u8,
    repo_id: []const u8,
    revision: []const u8,
) !ReadToken {
    return hub.requestReadToken(allocator, client, repo_type, repo_id, revision, auth);
}

/// Open a XET-backed file for lazy range reads from an already-resolved
/// hash/size and a (cached) read token. `hash_hexz` is copied; the caller keeps
/// ownership of `token` (the C-API only borrows it during `Session.init`).
pub fn openWith(
    allocator: std.mem.Allocator,
    hash_hexz: [:0]const u8,
    size: u64,
    token: hub.ReadToken,
) !RemoteFile {
    const hash = try allocator.dupeZ(u8, hash_hexz);
    errdefer allocator.free(hash);

    const session = try capi.Session.init(token.cas_url, token.access_token, token.exp);
    return .{ .session = session, .hash = hash, .size = size };
}
