//! Zig binding over xet-core's C-API (`xet_capi`, ABI from `hf_xet.h`).
//!
//! Drives the mature Rust xet client (adaptive concurrency, streaming,
//! decompress-once, on-disk chunk cache, BLAKE3 verification) instead of
//! reimplementing the protocol in Zig. Reconstruction runs at the same
//! throughput as the official `hf-xet` client.
//!
//! The C symbols are declared `extern` here rather than pulled in via
//! `@cImport`, so the build only needs to link `libxet_capi` (no header include
//! path to thread through the build system).
//!
//! A `Session` exposes two access modes against a CAS endpoint + token:
//!   - `downloadToPath`: reconstruct a whole file to disk (eager).
//!   - `readRange`: reconstruct just `[start, end)` into a buffer (lazy),
//!     used for positional VFS reads. Overlapping ranges hit the chunk cache.
//!
//! `HF_XET_HIGH_PERFORMANCE=1` enables peak concurrency/buffers, like `hf-xet`.
//! Auth (CAS endpoint + read token) comes from `xet_hub.requestReadToken`.

const std = @import("std");

const log = std.log.scoped(.@"zml/io/vfs/xet_capi");

// -- opaque handles --
const XetSession = opaque {};
const XetFileDownloadGroup = opaque {};
const XetDownloadStreamGroup = opaque {};
const XetDownloadStream = opaque {};
const XetFileInfo = opaque {};
const XetFileDownload = opaque {};
const XetOp = opaque {};
const XetError = opaque {};
const XetBytes = opaque {};
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
extern fn xet_session_new_download_stream_group(session: ?*XetSession, cfg: *const XetAuthConfig, out: *?*XetDownloadStreamGroup, err: *?*XetError) c_int;
extern fn xet_file_download_group_free(group: ?*XetFileDownloadGroup) void;
extern fn xet_download_stream_group_free(group: ?*XetDownloadStreamGroup) void;
extern fn xet_file_info_new(hash: [*:0]const u8, file_size: u64, out: *?*XetFileInfo, err: *?*XetError) c_int;
extern fn xet_file_info_free(fi: ?*XetFileInfo) void;
extern fn xet_file_download_group_download_to_path(group: ?*XetFileDownloadGroup, file_info: ?*XetFileInfo, dest_path: [*:0]const u8, out: *?*XetFileDownload, err: *?*XetError) c_int;
extern fn xet_file_download_group_finish_start(group: ?*XetFileDownloadGroup, out: *?*XetOp, err: *?*XetError) c_int;
extern fn xet_file_download_free(download: ?*XetFileDownload) void;
extern fn xet_download_stream_group_download_stream(group: ?*XetDownloadStreamGroup, file_info: ?*XetFileInfo, has_range: bool, range_start: u64, range_end: u64, out: *?*XetDownloadStream, err: *?*XetError) c_int;
extern fn xet_download_stream_next_start(stream: ?*XetDownloadStream, out: *?*XetOp, err: *?*XetError) c_int;
extern fn xet_download_stream_free(stream: ?*XetDownloadStream) void;
extern fn xet_op_poll(op: ?*XetOp) c_int;
extern fn xet_op_free(op: ?*XetOp) void;
extern fn xet_op_take_error(op: ?*XetOp, err: *?*XetError) c_int;
extern fn xet_op_take_bytes(op: ?*XetOp, out: *?*XetBytes, err: *?*XetError) c_int;
extern fn xet_op_take_download_report(op: ?*XetOp, out: *?*XetDownloadGroupReport, err: *?*XetError) c_int;
extern fn xet_download_group_report_free(r: ?*XetDownloadGroupReport) void;
extern fn xet_bytes_data(b: ?*XetBytes) [*]const u8;
extern fn xet_bytes_len(b: ?*XetBytes) usize;
extern fn xet_bytes_free(b: ?*XetBytes) void;
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

fn sleep1ms() void {
    var ts: std.c.timespec = .{ .sec = 0, .nsec = std.time.ns_per_ms };
    _ = std.c.nanosleep(&ts, null);
}

/// Block until `op` is ready, then take its bytes. Returns null at EOF.
fn awaitBytes(op: ?*XetOp) !?*XetBytes {
    while (true) {
        switch (xet_op_poll(op)) {
            XET_POLL_READY => {
                var bytes: ?*XetBytes = null;
                var err: ?*XetError = null;
                if (xet_op_take_bytes(op, &bytes, &err) != XET_OK) return capiError(err);
                return bytes;
            },
            XET_POLL_ERROR => {
                var err: ?*XetError = null;
                _ = xet_op_take_error(op, &err);
                return capiError(err);
            },
            else => sleep1ms(),
        }
    }
}

/// A live xet-core download session bound to one CAS endpoint + token.
pub const Session = struct {
    session: *XetSession,
    download_group: *XetFileDownloadGroup,
    stream_group: *XetDownloadStreamGroup,

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

        var download_group: ?*XetFileDownloadGroup = null;
        if (xet_session_new_file_download_group(session, &cfg, &download_group, &err) != XET_OK) return capiError(err);
        errdefer xet_file_download_group_free(download_group);

        var stream_group: ?*XetDownloadStreamGroup = null;
        if (xet_session_new_download_stream_group(session, &cfg, &stream_group, &err) != XET_OK) return capiError(err);

        return .{ .session = session.?, .download_group = download_group.?, .stream_group = stream_group.? };
    }

    pub fn deinit(self: *Session) void {
        xet_download_stream_group_free(self.stream_group);
        xet_file_download_group_free(self.download_group);
        xet_session_free(self.session);
    }

    /// Reconstruct the whole file (hex XET hash + size) to `dest_path`. Eager.
    pub fn downloadToPath(self: *Session, hash_hexz: [:0]const u8, size: u64, dest_pathz: [:0]const u8) !void {
        var err: ?*XetError = null;

        var fi: ?*XetFileInfo = null;
        if (xet_file_info_new(hash_hexz.ptr, size, &fi, &err) != XET_OK) return capiError(err);
        defer xet_file_info_free(fi);

        var download: ?*XetFileDownload = null;
        if (xet_file_download_group_download_to_path(self.download_group, fi, dest_pathz.ptr, &download, &err) != XET_OK) return capiError(err);
        defer xet_file_download_free(download);

        var op: ?*XetOp = null;
        if (xet_file_download_group_finish_start(self.download_group, &op, &err) != XET_OK) return capiError(err);
        defer xet_op_free(op);

        while (true) {
            switch (xet_op_poll(op)) {
                XET_POLL_READY => return,
                XET_POLL_ERROR => {
                    var e: ?*XetError = null;
                    _ = xet_op_take_error(op, &e);
                    return capiError(e);
                },
                else => sleep1ms(),
            }
        }
    }

    /// Reconstruct `[start, end)` of the file into `dest`, returning bytes
    /// written. Lazy: only the xorbs covering the range are fetched (overlapping
    /// ranges are served from the chunk cache). Used for positional reads.
    pub fn readRange(self: *Session, hash_hexz: [:0]const u8, size: u64, start: u64, end: u64, dest: []u8) !usize {
        var err: ?*XetError = null;

        var fi: ?*XetFileInfo = null;
        if (xet_file_info_new(hash_hexz.ptr, size, &fi, &err) != XET_OK) return capiError(err);
        defer xet_file_info_free(fi);

        var stream: ?*XetDownloadStream = null;
        if (xet_download_stream_group_download_stream(self.stream_group, fi, true, start, end, &stream, &err) != XET_OK) return capiError(err);
        defer xet_download_stream_free(stream);

        var written: usize = 0;
        while (written < dest.len) {
            var op: ?*XetOp = null;
            if (xet_download_stream_next_start(stream, &op, &err) != XET_OK) return capiError(err);

            const taken = awaitBytes(op);
            xet_op_free(op);
            const bytes = (taken catch |e| return e) orelse break; // null = EOF

            const src = xet_bytes_data(bytes)[0..xet_bytes_len(bytes)];
            const n = @min(src.len, dest.len - written);
            @memcpy(dest[written..][0..n], src[0..n]);
            written += n;
            xet_bytes_free(bytes);
        }
        return written;
    }
};
