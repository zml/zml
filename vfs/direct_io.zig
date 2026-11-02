//! Direct I/O on the descriptor of a local file. A VFS handle's `Mode` sets
//! the flag on its descriptor when a reader asks for it under a `Policy`,
//! and from then on issues that descriptor's positional reads here rather
//! than through the inner `std.Io`: the kernel answers an unsupported direct
//! read with `EINVAL` or `EFAULT`, which the threaded `std.Io` treats as a
//! programmer error, whereas here it is a refusal that clears the flag and
//! leaves the read to the VFS, buffered. Everything below `Mode` is Linux
//! only and reached from code gated on `supported`.
//!
//! Neither path wins everywhere, and residency alone cannot predict which
//! does: warm replicated Llama-3.1-8B on eight MI300X took ~1.10 s buffered
//! but ~4.53 s forced direct from a slow storage extent; on four GB300 with
//! four NVMe drives in RAID0, direct took ~0.30 s versus ~0.32 s warm
//! buffered. These are loader times, not disk-only rates.
const std = @import("std");
const builtin = @import("builtin");

const linux = std.os.linux;

const log = std.log.scoped(.@"zml/vfs");

pub const supported = builtin.target.os.tag == .linux and @hasField(std.posix.O, "DIRECT");

/// Alignment of a direct read's offset, every buffer and the total length.
/// 4 KiB covered the measured filesystems; this is not a statx(STATX_DIOALIGN)
/// query. A filesystem requiring more alignment may reject the first read,
/// after which the VFS demotes the handle to buffered.
pub const alignment: usize = 4 * 1024;

/// `Policy.auto` reads a file directly when at most this fraction of its
/// sampled pages is cached.
const cold_fraction: f64 = 0.5;

/// Whether a local file is read past the page cache. A direct read comes
/// from the disk even when the cache holds the data; it also leaves the cache
/// as it found it, so a file read directly is still cold for the next load
/// (the `auto` probe caches what it samples, elsewhere on every probe:
/// `cachedFraction`).
/// Raw warm-cache reads exceeded disk throughput in the measured setups,
/// but the loader can still favor direct reads: page-cache copies compete
/// with DMA for host memory bandwidth. Conversely, repeated loads from a
/// slow disk can benefit greatly from warming the cache with `off`. The
/// contrasting GB300/MI300X load results are in this file's header.
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
/// kernel rejected a read, or a read went through the inner `Io`. Each
/// method takes the inner `Io` and the handle's file on it.
pub const Mode = struct {
    /// Changes only under `mutex`, together with the flag; reads only load it.
    state: std.atomic.Value(State) = .init(.undecided),
    mutex: std.Io.Mutex = .init,

    const State = enum(u8) { undecided, buffered, direct };

    /// Whether `file` reads direct, deciding an undecided handle under
    /// `policy` (`VFS.useDirectIo`); `off` decides nothing.
    pub fn decide(self: *Mode, io: std.Io, file: std.Io.File, policy: Policy) bool {
        if (comptime !supported) return false;
        if (policy == .off) return false;
        switch (self.state.load(.acquire)) {
            .direct => return true,
            .buffered => return false,
            .undecided => {},
        }
        const stat = file.stat(io) catch return self.settle(io, file, false);
        if (stat.kind != .file or stat.size < alignment) return self.settle(io, file, false);
        if (policy == .auto) {
            const name: Name = .{ .io = io, .file = file };
            var shift: u64 = undefined;
            io.random(std.mem.asBytes(&shift));
            const cached = cachedFraction(file.handle, stat.size, shift) orelse {
                log.debug("{f}: page cache residency unknown; reading buffered", .{name});
                return self.settle(io, file, false);
            };
            if (cached > cold_fraction) {
                log.debug("{f}: {d:.0}% cached; reading buffered", .{ name, cached * 100 });
                return self.settle(io, file, false);
            }
            log.debug("{f}: {d:.0}% cached", .{ name, cached * 100 });
        }
        return self.settle(io, file, true);
    }

    /// Decides an undecided handle under the mutex, setting the flag first
    /// when `direct`, and returns whether the handle reads direct: a
    /// decision already taken (by a racing call, or a demotion) stands. A
    /// descriptor the flag cannot go on (`enable`) is read buffered.
    fn settle(self: *Mode, io: std.Io, file: std.Io.File, direct: bool) bool {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        switch (self.state.load(.acquire)) {
            .direct => return true,
            .buffered => return false,
            .undecided => {},
        }
        if (direct) {
            const name: Name = .{ .io = io, .file = file };
            const err = enable(file.handle);
            if (err == .SUCCESS) {
                self.state.store(.direct, .release);
                log.debug("{f}: reading direct", .{name});
                return true;
            }
            log.debug("{f}: O_DIRECT refused ({t}); reading buffered", .{ name, err });
        }
        self.state.store(.buffered, .release);
        return false;
    }

    /// The bytes read from `offset` into `data` past the page cache, or
    /// null for the caller to read them buffered: the handle is not direct,
    /// or the kernel refused the read and the handle is buffered from now
    /// on (`logRefusal`). A refusal demotes rather than fails: a legitimate
    /// short direct read can leave an unaligned continuation, and a
    /// filesystem accepting the flag does not prove it accepts these
    /// buffers (including PJRT-pinned memory) at read time.
    pub fn read(self: *Mode, io: std.Io, file: std.Io.File, data: []const []u8, offset: u64) std.Io.File.ReadPositionalError!?usize {
        if (comptime !supported) return null;
        if (self.state.load(.acquire) != .direct) return null;
        switch (try readPositional(file.handle, data, offset)) {
            .bytes => |bytes| return bytes,
            .refused => |err| {
                if (self.demote(io, file)) logRefusal(.{ .io = io, .file = file }, err, data, offset);
                return null;
            },
        }
    }

    /// Takes a direct handle back to buffered before a read the VFS does
    /// not issue itself (streaming, or as the source of a file copy): the
    /// inner `Io` treats the errno of a rejected direct read as a
    /// programmer error.
    pub fn leave(self: *Mode, io: std.Io, file: std.Io.File) void {
        if (comptime !supported) return;
        if (self.state.load(.acquire) != .direct or !self.demote(io, file)) return;
        const name: Name = .{ .io = io, .file = file };
        log.warn("{f}: buffered read of a direct file; reading it buffered from now on", .{name});
    }

    /// Takes a direct handle back to buffered: the flag comes off the
    /// descriptor before the state says buffered, under the mutex, so the
    /// inner `Io` never reads a descriptor with the flag on. True for the
    /// call that did it, which then says why.
    fn demote(self: *Mode, io: std.Io, file: std.Io.File) bool {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (self.state.load(.acquire) != .direct) return false;
        disable(file.handle);
        self.state.store(.buffered, .release);
        return true;
    }
};

/// A local file's path in a log line, resolved only when the line is
/// written.
const Name = struct {
    io: std.Io,
    file: std.Io.File,

    pub fn format(self: Name, writer: *std.Io.Writer) !void {
        var buffer: [std.fs.max_path_bytes]u8 = undefined;
        const len = self.file.realPath(self.io, &buffer) catch return writer.writeAll("<local file>");
        try writer.writeAll(buffer[0..len]);
    }
};

/// The kernel rejected a direct read. An aligned one is the filesystem
/// refusing direct I/O it accepted the flag for, a warning; a misaligned
/// one is the reader's, either its plan or a continuation after a short
/// read (which a network filesystem may answer), an error. Either way the
/// file is read buffered from now on.
fn logRefusal(name: Name, err: linux.E, data: []const []u8, offset: u64) void {
    var total: usize = 0;
    var aligned = offset % alignment == 0;
    for (data) |buffer| {
        if (buffer.len == 0) continue;
        aligned = aligned and @intFromPtr(buffer.ptr) % alignment == 0;
        total += buffer.len;
    }
    aligned = aligned and total % alignment == 0;
    if (aligned) {
        log.warn("{f}: direct read refused ({t}) at offset={d} total={d}; reading the file buffered from now on", .{ name, err, offset, total });
    } else {
        log.err("{f}: misaligned read of a direct file ({t}) at offset={d} first_buffer=0x{x} total={d} alignment={d}; reading the file buffered from now on", .{
            name,
            err,
            offset,
            if (data.len != 0) @intFromPtr(data[0].ptr) else 0,
            total,
            alignment,
        });
    }
}

const max_iovecs: usize = if (@TypeOf(std.posix.IOV_MAX) == void) 64 else std.posix.IOV_MAX;
const direct_flag: usize = @as(u32, @bitCast(std.posix.O{ .DIRECT = true }));

/// Sets the flag on a descriptor opened for reading only, since the flag
/// constrains writes too: one opened for writing is refused with `BADF`. A
/// filesystem without direct I/O refuses the flag.
pub fn enable(fd: std.posix.fd_t) linux.E {
    const current = switch (getStatusFlags(fd)) {
        .flags => |flags| flags,
        .err => |err| return err,
    };
    const status: std.posix.O = @bitCast(@as(u32, @truncate(current)));
    if (status.ACCMODE != .RDONLY) return .BADF;
    return setStatusFlags(fd, current | direct_flag);
}

/// Clears the flag, so the inner `Io` may read the descriptor again.
fn disable(fd: std.posix.fd_t) void {
    const current = switch (getStatusFlags(fd)) {
        .flags => |flags| flags,
        .err => return,
    };
    _ = setStatusFlags(fd, current & ~direct_flag);
}

/// `fcntl(F_GETFL)`, retried on `EINTR`.
fn getStatusFlags(fd: std.posix.fd_t) union(enum) { flags: usize, err: linux.E } {
    while (true) {
        const rc = linux.fcntl(fd, linux.F.GETFL, 0);
        switch (linux.errno(rc)) {
            .SUCCESS => return .{ .flags = rc },
            .INTR => continue,
            else => |err| return .{ .err = err },
        }
    }
}

/// `fcntl(F_SETFL)`, retried on `EINTR`.
fn setStatusFlags(fd: std.posix.fd_t, flags: usize) linux.E {
    while (true) {
        switch (linux.errno(linux.fcntl(fd, linux.F.SETFL, flags))) {
            .INTR => continue,
            else => |err| return err,
        }
    }
}

/// A fixed count bounds the probe's cost whatever the checkpoint's size and
/// suffices for the coarse half-cached decision. A whole-file `cachestat`
/// walks every cached folio, about 7 ms per cached GiB on a 4 KiB-page
/// kernel, paid on the planner thread, and refuses a file the process can
/// read but not write.
const residency_samples = 32;

/// The fraction of `residency_samples` pages spread over the file that the
/// page cache holds, or null when the filesystem cannot say (no
/// `RWF_NOWAIT`). One sample per window of the file, each `shift` bytes
/// into its window (`shift` modulo the window), unknown residency selecting
/// buffered IO. A sample that misses queues the read of its page, which a
/// direct file never reads back: with the same offsets the next probe would
/// find every sample cached and read a cold file buffered, so callers draw a
/// new `shift` for every probe. A file of at most `residency_samples` pages
/// has every page sampled whatever the shift, so its first probe caches it.
/// The header page a parser has just read is sampled only when `shift`
/// lands in it, not systematically.
fn cachedFraction(fd: std.posix.fd_t, size: u64, shift: u64) ?f64 {
    var buffer: [512]u8 = undefined;
    const iovec: std.posix.iovec = .{ .base = &buffer, .len = buffer.len };
    var cached: usize = 0;
    const window = size / residency_samples;
    for (0..residency_samples) |sample| {
        const offset = window * sample + shift % @max(window, 1);
        const rc = linux.preadv2(fd, @ptrCast(&iovec), 1, @intCast(offset), linux.RWF.NOWAIT);
        switch (linux.errno(rc)) {
            .SUCCESS => cached += @intFromBool(rc != 0),
            .AGAIN => {},
            else => return null,
        }
    }
    return @as(f64, @floatFromInt(cached)) / @as(f64, @floatFromInt(residency_samples));
}

const ReadResult = union(enum) {
    bytes: usize,
    /// The kernel rejected the direct read: an unaligned offset, buffer or
    /// total, a filesystem that took the flag but not the read, or memory
    /// the block layer cannot pin.
    refused: linux.E,
};

/// Reads `data` at `offset` on a descriptor with the flag set, at most
/// `IOV_MAX` buffers per call (the caller loops). A read that runs past the
/// end of the file is cut there.
fn readPositional(fd: std.posix.fd_t, data: []const []u8, offset: u64) std.Io.File.ReadPositionalError!ReadResult {
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
