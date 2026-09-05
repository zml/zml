//! Direct I/O on the descriptor of a local file. The VFS sets the flag on a
//! descriptor when a reader asks for it under a `Policy`, and from then on
//! issues that descriptor's positional reads here rather than through the
//! inner `std.Io`: the kernel answers an unsupported direct read with
//! `EINVAL` or `EFAULT`, which the threaded `std.Io` treats as a programmer
//! error, whereas here it is reported as a refusal for the VFS to clear the
//! flag and answer the read buffered. Everything below `State` is Linux
//! only and reached from code gated on `supported`.
const std = @import("std");
const builtin = @import("builtin");

const linux = std.os.linux;

pub const supported = builtin.target.os.tag == .linux and @hasField(std.posix.O, "DIRECT");

/// Alignment of a direct read's offset, every buffer and the total length.
/// 4 KiB satisfies every logical block size in use.
pub const alignment: usize = 4 * 1024;

/// `Policy.auto` reads a file directly when at most this fraction of its
/// sampled pages is cached.
pub const cold_fraction: f64 = 0.5;

/// Whether a local file is read past the page cache. A direct read comes
/// from the disk even when the cache holds the data, and a cached read is
/// several times faster than any disk; it also leaves the cache as it found
/// it, so a file read directly is still cold for the next load.
pub const Policy = enum {
    off,
    /// Every file the filesystem allows.
    on,
    /// A file mostly out of the page cache when asked, buffered for a cached
    /// one. Residency is sampled: `residency_samples` pages spread over the
    /// file, read with `RWF_NOWAIT`, which answers `EAGAIN` for a page that
    /// is not cached, needs no permission on the file and costs the same
    /// whatever its size. Unknown counts as cached.
    auto,
};

/// A local handle's read mode: undecided until a reader asks for direct
/// reads (`VFS.useDirectIo`), then buffered or direct for the rest of the
/// handle's life, since the flag belongs to the open file description and a
/// reader that was answered buffered plans exact reads that a direct
/// descriptor would reject. A direct handle goes back to buffered once the
/// kernel rejected a read, or a read went through the inner `Io`.
pub const State = enum(u8) { undecided, buffered, direct };

const max_iovecs: usize = if (@TypeOf(std.posix.IOV_MAX) == void) 64 else std.posix.IOV_MAX;
const direct_flag: usize = @as(u32, @bitCast(std.posix.O{ .DIRECT = true }));

/// Sets the flag; a filesystem without direct I/O refuses it.
pub fn enable(fd: std.posix.fd_t) linux.E {
    const current = linux.fcntl(fd, linux.F.GETFL, 0);
    const err = linux.errno(current);
    if (err != .SUCCESS) return err;
    return linux.errno(linux.fcntl(fd, linux.F.SETFL, current | direct_flag));
}

/// Clears the flag, so the inner `Io` may read the descriptor again.
pub fn disable(fd: std.posix.fd_t) void {
    const current = linux.fcntl(fd, linux.F.GETFL, 0);
    if (linux.errno(current) != .SUCCESS) return;
    _ = linux.fcntl(fd, linux.F.SETFL, current & ~direct_flag);
}

pub const residency_samples = 32;

/// The fraction of `residency_samples` pages spread over the file that the
/// page cache holds, or null when the filesystem cannot say (no
/// `RWF_NOWAIT`). Each sample sits mid-window, so the first one is not the
/// header page a parser has just read. A sample of a page that is not
/// cached queues readahead for it, a few pages per sample the reads that
/// follow may reuse.
pub fn cachedFraction(fd: std.posix.fd_t, size: u64) ?f64 {
    var buffer: [512]u8 = undefined;
    const iovec: std.posix.iovec = .{ .base = &buffer, .len = buffer.len };
    var cached: usize = 0;
    const window = size / residency_samples;
    for (0..residency_samples) |sample| {
        const offset = window * sample + window / 2;
        const rc = linux.preadv2(fd, @ptrCast(&iovec), 1, @intCast(offset), linux.RWF.NOWAIT);
        switch (linux.errno(rc)) {
            .SUCCESS => cached += @intFromBool(rc != 0),
            .AGAIN => {},
            else => return null,
        }
    }
    return @as(f64, @floatFromInt(cached)) / @as(f64, @floatFromInt(residency_samples));
}

pub const ReadResult = union(enum) {
    bytes: usize,
    /// The kernel rejected the direct read: an unaligned offset, buffer or
    /// total, a filesystem that took the flag but not the read, or memory
    /// the block layer cannot pin.
    refused: linux.E,
};

/// Reads `data` at `offset` on a descriptor with the flag set, at most
/// `IOV_MAX` buffers per call (the caller loops). A read that runs past the
/// end of the file is cut there.
pub fn readPositional(fd: std.posix.fd_t, data: []const []u8, offset: u64) std.Io.File.ReadPositionalError!ReadResult {
    var iovecs: [max_iovecs]std.posix.iovec = undefined;
    var count: usize = 0;
    for (data) |buffer| {
        if (buffer.len == 0) continue;
        if (count == iovecs.len) break;
        iovecs[count] = .{ .base = buffer.ptr, .len = buffer.len };
        count += 1;
    }
    if (count == 0) return .{ .bytes = 0 };
    while (true) {
        const rc = linux.preadv(fd, &iovecs, count, @intCast(offset));
        switch (linux.errno(rc)) {
            .SUCCESS => return .{ .bytes = rc },
            .INTR => continue,
            .INVAL, .FAULT, .OPNOTSUPP => |err| return .{ .refused = err },
            .IO => return error.InputOutput,
            .NOMEM => return error.SystemResources,
            .AGAIN => return error.WouldBlock,
            .ISDIR => return error.IsDir,
            .BADF => return error.NotOpenForReading,
            .NXIO, .SPIPE => return error.Unseekable,
            else => |err| return std.posix.unexpectedErrno(err),
        }
    }
}
