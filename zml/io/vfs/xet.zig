//! Native XET download for ZML, backed by xet-core (Rust) through its C-API.
//!
//! Rather than reimplementing the XET protocol in Zig, this drives the mature
//! xet-core client via `xet_capi.zig`. You get the same throughput as the
//! official `hf-xet` client (adaptive concurrency, streaming, decompress-once,
//! on-disk chunk cache, BLAKE3 verification).
//!
//! Two entry points:
//!   - `downloadFile`: reconstruct a whole repo file to a path (eager).
//!   - `openRemote`: resolve a repo file to a reusable `RemoteFile` whose
//!     `readRange` reconstructs arbitrary byte ranges on demand (lazy). This is
//!     what `hf.zig` uses to serve positional reads of `hf://` XET files.
//!
//! Set `HF_XET_HIGH_PERFORMANCE=1` for peak concurrency/buffers.

const std = @import("std");

pub const hub = @import("xet_hub.zig");
pub const capi = @import("xet_capi.zig");

pub const Session = capi.Session;

const log = std.log.scoped(.@"zml/io/vfs/xet");

fn bearer(allocator: std.mem.Allocator, hf_token: []const u8) ![]const u8 {
    return std.fmt.allocPrint(allocator, "Bearer {s}", .{hf_token});
}

/// Look up `filepath` among the repo's XET-backed files. Caller owns the
/// returned hash (NUL-terminated) and gets the file size.
fn resolveHash(
    allocator: std.mem.Allocator,
    client: *std.http.Client,
    repo_type: []const u8,
    repo_id: []const u8,
    revision: []const u8,
    filepath: []const u8,
    auth: hub.Auth,
) !struct { hash: [:0]const u8, size: u64 } {
    const files = try hub.listXetFiles(allocator, client, repo_type, repo_id, revision, auth);
    defer {
        for (files) |f| {
            allocator.free(f.path);
            allocator.free(f.hash_hexz);
        }
        allocator.free(files);
    }
    for (files) |f| {
        if (std.mem.eql(u8, f.path, filepath)) {
            return .{ .hash = try allocator.dupeZ(u8, f.hash_hexz), .size = f.size };
        }
    }
    return error.FileNotXetBacked;
}

/// A repo file resolved to a live xet-core session, ready for range reads.
pub const RemoteFile = struct {
    session: Session,
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

/// Open a XET-backed repo file for lazy range reads. Errors (leaving nothing to
/// clean up) if the file is not XET-backed, so callers can fall back.
pub fn openRemote(
    allocator: std.mem.Allocator,
    io: std.Io,
    repo_type: []const u8,
    repo_id: []const u8,
    revision: []const u8,
    filepath: []const u8,
    hf_token: []const u8,
) !RemoteFile {
    var client = std.http.Client{ .allocator = allocator, .io = io };
    defer client.deinit();

    const auth_value = try bearer(allocator, hf_token);
    defer allocator.free(auth_value);
    const auth: hub.Auth = .{ .override = auth_value };

    const file = try resolveHash(allocator, &client, repo_type, repo_id, revision, filepath, auth);
    errdefer allocator.free(file.hash);

    const token = try hub.requestReadToken(allocator, &client, repo_type, repo_id, revision, auth);
    defer token.deinit(allocator); // C-API only borrows these during init

    const session = try Session.init(token.cas_url, token.access_token, token.exp);
    return .{ .session = session, .hash = file.hash, .size = file.size };
}

/// Download a whole XET-backed repo file to `dest_path` (eager).
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
    var remote = try openRemote(allocator, io, repo_type, repo_id, revision, filepath, hf_token);
    defer remote.deinit(allocator);
    try remote.session.downloadToPath(remote.hash, remote.size, dest_path);
}
