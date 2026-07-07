//! Zig binding over xet-core's C-API (`xet_capi`, ABI from `hf_xet.h`).
//!
//! This drives the mature Rust xet client (adaptive concurrency, streaming,
//! decompress-once, on-disk chunk cache, BLAKE3 verification) instead of
//! reimplementing the protocol in Zig. Reconstruction runs at the same
//! throughput as the official `hf-xet` client.
//!
//! The C symbols are declared `extern` here rather than pulled in via
//! `@cImport`, so the build only needs to link `libxet_capi` (no header include
//! path to thread through the build system).
//!
//! High-performance mode (bigger concurrency + buffers) is enabled by setting
//! `HF_XET_HIGH_PERFORMANCE=1` before creating a `Session`, like `hf-xet`.
//! Auth (CAS endpoint + read token) comes from `xet_hub.requestReadToken`.

const std = @import("std");

const log = std.log.scoped(.@"zml/io/vfs/xet_capi");

// -- opaque handles --
const XetSession = opaque {};
const XetFileDownloadGroup = opaque {};
const XetFileInfo = opaque {};
const XetFileDownload = opaque {};
const XetOp = opaque {};
const XetError = opaque {};
const XetDownloadGroupReport = opaque {};

const XetHeader = extern struct {
    key: ?[*:0]const u8,
    value: ?[*:0]const u8,
};

const XetAuthConfig = extern struct {
    endpoint: ?[*:0]const u8,
    token: ?[*:0]const u8,
    token_expiry: i64,
    token_refresh_url: ?[*:0]const u8,
    refresh_headers: ?[*]const XetHeader,
    refresh_header_count: usize,
};

// XetStatus: 0 = Ok. XetPollState: 0 = Pending, 1 = Ready, 2 = Error.
const XET_OK: c_int = 0;
const XET_POLL_READY: c_int = 1;
const XET_POLL_ERROR: c_int = 2;

extern fn xet_session_new(out: *?*XetSession, err: *?*XetError) c_int;
extern fn xet_session_free(session: ?*XetSession) void;
extern fn xet_session_new_file_download_group(session: ?*XetSession, cfg: *const XetAuthConfig, out: *?*XetFileDownloadGroup, err: *?*XetError) c_int;
extern fn xet_file_download_group_free(group: ?*XetFileDownloadGroup) void;
extern fn xet_file_info_new(hash: [*:0]const u8, file_size: u64, out: *?*XetFileInfo, err: *?*XetError) c_int;
extern fn xet_file_info_free(fi: ?*XetFileInfo) void;
extern fn xet_file_download_group_download_to_path(group: ?*XetFileDownloadGroup, file_info: ?*XetFileInfo, dest_path: [*:0]const u8, out: *?*XetFileDownload, err: *?*XetError) c_int;
extern fn xet_file_download_group_finish_start(group: ?*XetFileDownloadGroup, out: *?*XetOp, err: *?*XetError) c_int;
extern fn xet_file_download_free(download: ?*XetFileDownload) void;
extern fn xet_op_poll(op: ?*XetOp) c_int;
extern fn xet_op_free(op: ?*XetOp) void;
extern fn xet_op_take_error(op: ?*XetOp, err: *?*XetError) c_int;
extern fn xet_op_take_download_report(op: ?*XetOp, out: *?*XetDownloadGroupReport, err: *?*XetError) c_int;
extern fn xet_download_group_report_free(r: ?*XetDownloadGroupReport) void;
extern fn xet_error_message(err: ?*XetError) [*:0]const u8;
extern fn xet_error_code(err: ?*XetError) c_int;
extern fn xet_error_free(err: ?*XetError) void;

fn capiError(err: ?*XetError) error{XetCapi} {
    if (err) |e| {
        log.err("xet_capi: {s} (code {d})", .{ xet_error_message(e), xet_error_code(e) });
        xet_error_free(e);
    }
    return error.XetCapi;
}

/// A live xet-core download session bound to one CAS endpoint + token.
pub const Session = struct {
    session: *XetSession,
    group: *XetFileDownloadGroup,

    pub fn init(cas_url: [:0]const u8, token: [:0]const u8, token_expiry: i64) !Session {
        var err: ?*XetError = null;

        var session: ?*XetSession = null;
        if (xet_session_new(&session, &err) != XET_OK) return capiError(err);
        errdefer xet_session_free(session);

        const cfg = XetAuthConfig{
            .endpoint = cas_url.ptr,
            .token = token.ptr,
            .token_expiry = token_expiry,
            .token_refresh_url = null,
            .refresh_headers = null,
            .refresh_header_count = 0,
        };

        var group: ?*XetFileDownloadGroup = null;
        if (xet_session_new_file_download_group(session, &cfg, &group, &err) != XET_OK) return capiError(err);

        return .{ .session = session.?, .group = group.? };
    }

    pub fn deinit(self: *Session) void {
        xet_file_download_group_free(self.group);
        xet_session_free(self.session);
    }

    /// Download the file identified by its hex XET hash + size to `dest_path`.
    /// Blocks until the reconstruction completes.
    pub fn downloadToPath(self: *Session, hash_hexz: [:0]const u8, size: u64, dest_pathz: [:0]const u8) !void {
        var err: ?*XetError = null;

        var fi: ?*XetFileInfo = null;
        if (xet_file_info_new(hash_hexz.ptr, size, &fi, &err) != XET_OK) return capiError(err);
        defer xet_file_info_free(fi);

        var download: ?*XetFileDownload = null;
        if (xet_file_download_group_download_to_path(self.group, fi, dest_pathz.ptr, &download, &err) != XET_OK) return capiError(err);
        defer xet_file_download_free(download);

        var op: ?*XetOp = null;
        if (xet_file_download_group_finish_start(self.group, &op, &err) != XET_OK) return capiError(err);
        defer xet_op_free(op);

        while (true) {
            switch (xet_op_poll(op)) {
                XET_POLL_READY => return,
                XET_POLL_ERROR => {
                    var e: ?*XetError = null;
                    _ = xet_op_take_error(op, &e);
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
