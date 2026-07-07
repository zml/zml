//! Zig binding over xet-core's C-API (`xet_capi`, header `hf_xet.h`).
//!
//! This drives the mature Rust xet client (adaptive concurrency, streaming,
//! decompress-once, on-disk chunk cache, BLAKE3 verification) instead of
//! reimplementing the protocol in Zig. Reconstruction runs at the same
//! throughput as the official `hf-xet` client.
//!
//! High-performance mode (bigger concurrency + buffers) is enabled by setting
//! the `HF_XET_HIGH_PERFORMANCE=1` environment variable before creating a
//! `Session`, exactly like `hf-xet`.
//!
//! Auth (CAS endpoint + read token) comes from `xet_hub.requestReadToken`.

const std = @import("std");

const c = @cImport({
    @cInclude("hf_xet.h");
});

const log = std.log.scoped(.@"zml/io/vfs/xet_capi");

fn capiError(err: ?*c.XetError) error{XetCapi} {
    if (err) |e| {
        log.err("xet_capi: {s} (code {d})", .{ c.xet_error_message(e), c.xet_error_code(e) });
        c.xet_error_free(e);
    }
    return error.XetCapi;
}

/// A live xet-core download session bound to one CAS endpoint + token.
pub const Session = struct {
    session: *c.XetSession,
    group: *c.XetFileDownloadGroup,

    pub fn init(cas_url: [:0]const u8, token: [:0]const u8, token_expiry: i64) !Session {
        var err: ?*c.XetError = null;

        var session: ?*c.XetSession = null;
        if (c.xet_session_new(&session, &err) != c.XetStatus_XetOk) return capiError(err);
        errdefer c.xet_session_free(session);

        var cfg = c.XetAuthConfig{
            .endpoint = cas_url.ptr,
            .token = token.ptr,
            .token_expiry = token_expiry,
            .token_refresh_url = null,
            .refresh_headers = null,
            .refresh_header_count = 0,
        };

        var group: ?*c.XetFileDownloadGroup = null;
        if (c.xet_session_new_file_download_group(session, &cfg, &group, &err) != c.XetStatus_XetOk) {
            return capiError(err);
        }

        return .{ .session = session.?, .group = group.? };
    }

    pub fn deinit(self: *Session) void {
        c.xet_file_download_group_free(self.group);
        c.xet_session_free(self.session);
    }

    /// Download the file identified by its hex XET hash + size to `dest_path`.
    /// Blocks until the reconstruction completes.
    pub fn downloadToPath(self: *Session, hash_hexz: [:0]const u8, size: u64, dest_pathz: [:0]const u8) !void {
        var err: ?*c.XetError = null;

        var fi: ?*c.XetFileInfo = null;
        if (c.xet_file_info_new(hash_hexz.ptr, size, &fi, &err) != c.XetStatus_XetOk) return capiError(err);
        defer c.xet_file_info_free(fi);

        var download: ?*c.XetFileDownload = null;
        if (c.xet_file_download_group_download_to_path(self.group, fi, dest_pathz.ptr, &download, &err) != c.XetStatus_XetOk) {
            return capiError(err);
        }
        defer if (download) |d| c.xet_file_download_free(d);

        var op: ?*c.XetOp = null;
        if (c.xet_file_download_group_finish_start(self.group, &op, &err) != c.XetStatus_XetOk) return capiError(err);
        defer c.xet_op_free(op);

        while (true) {
            switch (c.xet_op_poll(op)) {
                c.XetPollState_XetPollReady => return,
                c.XetPollState_XetPollError => {
                    var e: ?*c.XetError = null;
                    _ = c.xet_op_take_error(op, &e);
                    return capiError(e);
                },
                else => {
                    var ts: std.c.timespec = .{ .sec = 0, .nsec = std.time.ns_per_ms };
                    _ = std.c.nanosleep(&ts, null);
                },
            }
        }
    }
};
