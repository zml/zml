//! Native XET reads for ZML, backed by xet-core (Rust) through its C-API.
//!
//! Rather than reimplementing the XET protocol in Zig, this drives the mature
//! xet-core client via `xet_capi.zig`. You get the same throughput as the
//! official `hf-xet` client (adaptive concurrency, streaming, decompress-once,
//! on-disk chunk cache, BLAKE3 verification).
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

/// Open a XET-backed repo file for lazy range reads, reusing the caller's HTTP
/// client and authorization. Returns `error.FileNotXetBacked` (with nothing to
/// clean up) if the file is not XET-backed, so callers can fall back.
pub fn openRemote(
    allocator: std.mem.Allocator,
    client: *std.http.Client,
    auth: hub.Auth,
    repo_type: []const u8,
    repo_id: []const u8,
    revision: []const u8,
    filepath: []const u8,
) !RemoteFile {
    const file = try hub.findXetFile(allocator, client, repo_type, repo_id, revision, filepath, auth);
    errdefer allocator.free(file.hash_hexz);

    const token = try hub.requestReadToken(allocator, client, repo_type, repo_id, revision, auth);
    defer token.deinit(allocator); // C-API only borrows these during init

    const session = try capi.Session.init(token.cas_url, token.access_token, token.exp);
    return .{ .session = session, .hash = file.hash_hexz, .size = file.size };
}
