const std = @import("std");

const stdx = @import("stdx");

pub const VFSBase = struct {
    pub fn HandleRegistry(comptime InnerType: type) type {
        return struct {
            const Self = @This();

            const Node = struct {
                value: InnerType,
                next_closed: ?usize = null,
            };

            mutex: std.Io.Mutex = .init,
            handles: stdx.SegmentedList(InnerType, 0) = .{},
            next_closed: ?usize = null,

            pub fn open(self: *Self, allocator: std.mem.Allocator, io: std.Io) !struct { usize, *InnerType } {
                self.mutex.lockUncancelable(io);
                defer self.mutex.unlock(io);

                if (self.next_closed) |handle| {
                    const node = self.handles.at(handle);
                    self.next_closed = node.next_closed;
                    return .{ handle, node.value };
                }
                return .{ self.handles.count(), try self.handles.addOne(allocator) };
            }

            pub fn close(self: *Self, io: std.Io, handle: usize) void {
                self.mutex.lockUncancelable(io);
                defer self.mutex.unlock(io);

                const node = self.handles.at(handle);
                node.next_closed = self.next_closed;
                self.next_closed = handle;
            }

            pub fn get(self: *Self, io: std.Io, handle: usize) *InnerType {
                self.mutex.lockUncancelable(io);
                defer self.mutex.unlock(io);

                return &self.handles.at(handle).value;
            }
        };
    }

    inner: std.Io,

    pub fn init(io: std.Io) VFSBase {
        return .{ .inner = io };
    }

    pub fn vtable(overrides: anytype) std.Io.VTable {
        var new_vtable: std.Io.VTable = .{
            .async = async,
            .concurrent = concurrent,
            .await = await,
            .cancel = cancel,
            .groupAsync = groupAsync,
            .groupConcurrent = groupConcurrent,
            .groupAwait = groupAwait,
            .groupCancel = groupCancel,
            .recancel = recancel,
            .swapCancelProtection = swapCancelProtection,
            .checkCancel = checkCancel,
            .select = select,
            .futexWait = futexWait,
            .futexWaitUncancelable = futexWaitUncancelable,
            .futexWake = futexWake,
            .dirCreateDir = dirCreateDir,
            .dirCreateDirPath = dirCreateDirPath,
            .dirCreateDirPathOpen = dirCreateDirPathOpen,
            .dirOpenDir = dirOpenDir,
            .dirStat = dirStat,
            .dirStatFile = dirStatFile,
            .dirAccess = dirAccess,
            .dirCreateFile = dirCreateFile,
            .dirOpenFile = dirOpenFile,
            .dirClose = dirClose,
            .dirRead = dirRead,
            .dirRealPath = dirRealPath,
            .dirRealPathFile = dirRealPathFile,
            .dirDeleteFile = dirDeleteFile,
            .dirDeleteDir = dirDeleteDir,
            .dirRename = dirRename,
            .dirSymLink = dirSymLink,
            .dirReadLink = dirReadLink,
            .dirSetOwner = dirSetOwner,
            .dirSetFileOwner = dirSetFileOwner,
            .dirSetPermissions = dirSetPermissions,
            .dirSetFilePermissions = dirSetFilePermissions,
            .dirSetTimestamps = dirSetTimestamps,
            .dirHardLink = dirHardLink,
            .fileStat = fileStat,
            .fileLength = fileLength,
            .fileClose = fileClose,
            .fileWriteStreaming = fileWriteStreaming,
            .fileWritePositional = fileWritePositional,
            .fileWriteFileStreaming = fileWriteFileStreaming,
            .fileWriteFilePositional = fileWriteFilePositional,
            .fileReadStreaming = fileReadStreaming,
            .fileReadPositional = fileReadPositional,
            .fileSeekBy = fileSeekBy,
            .fileSeekTo = fileSeekTo,
            .fileSync = fileSync,
            .fileIsTty = fileIsTty,
            .fileEnableAnsiEscapeCodes = fileEnableAnsiEscapeCodes,
            .fileSupportsAnsiEscapeCodes = fileSupportsAnsiEscapeCodes,
            .fileSetLength = fileSetLength,
            .fileSetOwner = fileSetOwner,
            .fileSetPermissions = fileSetPermissions,
            .fileSetTimestamps = fileSetTimestamps,
            .fileLock = fileLock,
            .fileTryLock = fileTryLock,
            .fileUnlock = fileUnlock,
            .fileDowngradeLock = fileDowngradeLock,
            .fileRealPath = fileRealPath,
            .processExecutableOpen = processExecutableOpen,
            .processExecutablePath = processExecutablePath,
            .lockStderr = lockStderr,
            .tryLockStderr = tryLockStderr,
            .unlockStderr = unlockStderr,
            .processSetCurrentDir = processSetCurrentDir,
            .now = now,
            .sleep = sleep,
            .netListenIp = netListenIp,
            .netAccept = netAccept,
            .netBindIp = netBindIp,
            .netConnectIp = netConnectIp,
            .netListenUnix = netListenUnix,
            .netConnectUnix = netConnectUnix,
            .netSend = netSend,
            .netReceive = netReceive,
            .netRead = netRead,
            .netWrite = netWrite,
            .netWriteFile = netWriteFile,
            .netClose = netClose,
            .netShutdown = netShutdown,
            .netInterfaceNameResolve = netInterfaceNameResolve,
            .netInterfaceName = netInterfaceName,
            .netLookup = netLookup,
        };
        for (std.meta.fieldNames(@TypeOf(overrides))) |field_name| {
            @field(new_vtable, field_name) = @field(overrides, field_name);
        }
        return new_vtable;
    }

    pub fn as(userdata: ?*anyopaque) *VFSBase {
        return @ptrCast(@alignCast(userdata.?));
    }

    pub fn async(userdata: ?*anyopaque, result: []u8, result_alignment: std.mem.Alignment, context: []const u8, context_alignment: std.mem.Alignment, start: *const fn (context: *const anyopaque, result: *anyopaque) void) ?*std.Io.AnyFuture {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.async(self.inner.userdata, result, result_alignment, context, context_alignment, start);
    }

    pub fn concurrent(userdata: ?*anyopaque, result_len: usize, result_alignment: std.mem.Alignment, context: []const u8, context_alignment: std.mem.Alignment, start: *const fn (context: *const anyopaque, result: *anyopaque) void) std.Io.ConcurrentError!*std.Io.AnyFuture {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.concurrent(self.inner.userdata, result_len, result_alignment, context, context_alignment, start);
    }

    pub fn await(userdata: ?*anyopaque, any_future: *std.Io.AnyFuture, result: []u8, result_alignment: std.mem.Alignment) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.await(self.inner.userdata, any_future, result, result_alignment);
    }

    pub fn cancel(userdata: ?*anyopaque, any_future: *std.Io.AnyFuture, result: []u8, result_alignment: std.mem.Alignment) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.cancel(self.inner.userdata, any_future, result, result_alignment);
    }

    pub fn groupAsync(userdata: ?*anyopaque, group: *std.Io.Group, context: []const u8, context_alignment: std.mem.Alignment, start: *const fn (context: *const anyopaque) std.Io.Cancelable!void) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.groupAsync(self.inner.userdata, group, context, context_alignment, start);
    }

    pub fn groupConcurrent(userdata: ?*anyopaque, group: *std.Io.Group, context: []const u8, context_alignment: std.mem.Alignment, start: *const fn (context: *const anyopaque) std.Io.Cancelable!void) std.Io.ConcurrentError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.groupConcurrent(self.inner.userdata, group, context, context_alignment, start);
    }

    pub fn groupAwait(userdata: ?*anyopaque, group: *std.Io.Group, token: *anyopaque) std.Io.Cancelable!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.groupAwait(self.inner.userdata, group, token);
    }

    pub fn groupCancel(userdata: ?*anyopaque, group: *std.Io.Group, token: *anyopaque) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.groupCancel(self.inner.userdata, group, token);
    }

    pub fn recancel(userdata: ?*anyopaque) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.recancel(self.inner.userdata);
    }

    pub fn swapCancelProtection(userdata: ?*anyopaque, new: std.Io.CancelProtection) std.Io.CancelProtection {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.swapCancelProtection(self.inner.userdata, new);
    }

    pub fn checkCancel(userdata: ?*anyopaque) std.Io.Cancelable!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.checkCancel(self.inner.userdata);
    }

    pub fn select(userdata: ?*anyopaque, futures: []const *std.Io.AnyFuture) std.Io.Cancelable!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.select(self.inner.userdata, futures);
    }

    pub fn futexWait(userdata: ?*anyopaque, ptr: *const u32, expected: u32, timeout: std.Io.Timeout) std.Io.Cancelable!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.futexWait(self.inner.userdata, ptr, expected, timeout);
    }

    pub fn futexWaitUncancelable(userdata: ?*anyopaque, ptr: *const u32, expected: u32) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.futexWaitUncancelable(self.inner.userdata, ptr, expected);
    }

    pub fn futexWake(userdata: ?*anyopaque, ptr: *const u32, max_waiters: u32) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.futexWake(self.inner.userdata, ptr, max_waiters);
    }

    pub fn dirCreateDir(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, permissions: std.Io.Dir.Permissions) std.Io.Dir.CreateDirError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirCreateDir(self.inner.userdata, dir, sub_path, permissions);
    }

    pub fn dirCreateDirPath(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, permissions: std.Io.Dir.Permissions) std.Io.Dir.CreateDirPathError!std.Io.Dir.CreatePathStatus {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirCreateDirPath(self.inner.userdata, dir, sub_path, permissions);
    }

    pub fn dirCreateDirPathOpen(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, permissions: std.Io.Dir.Permissions, options: std.Io.Dir.OpenOptions) std.Io.Dir.CreateDirPathOpenError!std.Io.Dir {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirCreateDirPathOpen(self.inner.userdata, dir, sub_path, permissions, options);
    }

    pub fn dirOpenDir(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, options: std.Io.Dir.OpenOptions) std.Io.Dir.OpenError!std.Io.Dir {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirOpenDir(self.inner.userdata, dir, sub_path, options);
    }

    pub fn dirStat(userdata: ?*anyopaque, dir: std.Io.Dir) std.Io.Dir.StatError!std.Io.Dir.Stat {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirStat(self.inner.userdata, dir);
    }

    pub fn dirStatFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, options: std.Io.Dir.StatFileOptions) std.Io.Dir.StatFileError!std.Io.File.Stat {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirStatFile(self.inner.userdata, dir, sub_path, options);
    }

    pub fn dirAccess(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, options: std.Io.Dir.AccessOptions) std.Io.Dir.AccessError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirAccess(self.inner.userdata, dir, sub_path, options);
    }

    pub fn dirCreateFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, flags: std.Io.File.CreateFlags) std.Io.File.OpenError!std.Io.File {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirCreateFile(self.inner.userdata, dir, sub_path, flags);
    }

    pub fn dirOpenFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, flags: std.Io.File.OpenFlags) std.Io.File.OpenError!std.Io.File {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirOpenFile(self.inner.userdata, dir, sub_path, flags);
    }

    pub fn dirClose(userdata: ?*anyopaque, dirs: []const std.Io.Dir) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        self.inner.vtable.dirClose(self.inner.userdata, dirs);
    }

    pub fn dirRead(userdata: ?*anyopaque, reader: *std.Io.Dir.Reader, entries: []std.Io.Dir.Entry) std.Io.Dir.Reader.Error!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirRead(self.inner.userdata, reader, entries);
    }

    pub fn dirRealPath(userdata: ?*anyopaque, dir: std.Io.Dir, out_buffer: []u8) std.Io.Dir.RealPathError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirRealPath(self.inner.userdata, dir, out_buffer);
    }

    pub fn dirRealPathFile(userdata: ?*anyopaque, dir: std.Io.Dir, path_name: []const u8, out_buffer: []u8) std.Io.Dir.RealPathFileError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirRealPathFile(self.inner.userdata, dir, path_name, out_buffer);
    }

    pub fn dirDeleteFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8) std.Io.Dir.DeleteFileError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirDeleteFile(self.inner.userdata, dir, sub_path);
    }

    pub fn dirDeleteDir(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8) std.Io.Dir.DeleteDirError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirDeleteDir(self.inner.userdata, dir, sub_path);
    }

    pub fn dirRename(userdata: ?*anyopaque, old_dir: std.Io.Dir, old_sub_path: []const u8, new_dir: std.Io.Dir, new_sub_path: []const u8) std.Io.Dir.RenameError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirRename(self.inner.userdata, old_dir, old_sub_path, new_dir, new_sub_path);
    }

    pub fn dirSymLink(userdata: ?*anyopaque, dir: std.Io.Dir, target_path: []const u8, sym_link_path: []const u8, flags: std.Io.Dir.SymLinkFlags) std.Io.Dir.SymLinkError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirSymLink(self.inner.userdata, dir, target_path, sym_link_path, flags);
    }

    pub fn dirReadLink(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, buffer: []u8) std.Io.Dir.ReadLinkError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirReadLink(self.inner.userdata, dir, sub_path, buffer);
    }

    pub fn dirSetOwner(userdata: ?*anyopaque, dir: std.Io.Dir, uid: ?std.Io.File.Uid, gid: ?std.Io.File.Gid) std.Io.Dir.SetOwnerError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirSetOwner(self.inner.userdata, dir, uid, gid);
    }

    pub fn dirSetFileOwner(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, uid: ?std.Io.File.Uid, gid: ?std.Io.File.Gid, options: std.Io.Dir.SetFileOwnerOptions) std.Io.Dir.SetFileOwnerError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirSetFileOwner(self.inner.userdata, dir, sub_path, uid, gid, options);
    }

    pub fn dirSetPermissions(userdata: ?*anyopaque, dir: std.Io.Dir, permissions: std.Io.Dir.Permissions) std.Io.Dir.SetPermissionsError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirSetPermissions(self.inner.userdata, dir, permissions);
    }

    pub fn dirSetFilePermissions(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, permissions: std.Io.File.Permissions, options: std.Io.Dir.SetFilePermissionsOptions) std.Io.Dir.SetFilePermissionsError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirSetFilePermissions(self.inner.userdata, dir, sub_path, permissions, options);
    }

    pub fn dirSetTimestamps(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, options: std.Io.Dir.SetTimestampsOptions) std.Io.Dir.SetTimestampsError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirSetTimestamps(self.inner.userdata, dir, sub_path, options);
    }

    pub fn dirHardLink(userdata: ?*anyopaque, old_dir: std.Io.Dir, old_sub_path: []const u8, new_dir: std.Io.Dir, new_sub_path: []const u8, options: std.Io.Dir.HardLinkOptions) std.Io.Dir.HardLinkError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirHardLink(self.inner.userdata, old_dir, old_sub_path, new_dir, new_sub_path, options);
    }

    pub fn fileStat(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.StatError!std.Io.File.Stat {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileStat(self.inner.userdata, file);
    }

    pub fn fileLength(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.LengthError!u64 {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileLength(self.inner.userdata, file);
    }

    pub fn fileClose(userdata: ?*anyopaque, files: []const std.Io.File) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        self.inner.vtable.fileClose(self.inner.userdata, files);
    }

    pub fn fileWriteStreaming(userdata: ?*anyopaque, file: std.Io.File, header: []const u8, data: []const []const u8, splat: usize) std.Io.File.Writer.Error!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileWriteStreaming(self.inner.userdata, file, header, data, splat);
    }

    pub fn fileWritePositional(userdata: ?*anyopaque, file: std.Io.File, header: []const u8, data: []const []const u8, splat: usize, offset: u64) std.Io.File.WritePositionalError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileWritePositional(self.inner.userdata, file, header, data, splat, offset);
    }

    pub fn fileWriteFileStreaming(userdata: ?*anyopaque, file: std.Io.File, header: []const u8, reader: *std.Io.File.Reader, limit: std.Io.Limit) std.Io.File.Writer.WriteFileError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileWriteFileStreaming(self.inner.userdata, file, header, reader, limit);
    }

    pub fn fileWriteFilePositional(userdata: ?*anyopaque, file: std.Io.File, header: []const u8, reader: *std.Io.File.Reader, limit: std.Io.Limit, offset: u64) std.Io.File.WriteFilePositionalError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileWriteFilePositional(self.inner.userdata, file, header, reader, limit, offset);
    }

    pub fn fileReadStreaming(userdata: ?*anyopaque, file: std.Io.File, data: []const []u8) std.Io.File.Reader.Error!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileReadStreaming(self.inner.userdata, file, data);
    }

    pub fn fileReadPositional(userdata: ?*anyopaque, file: std.Io.File, data: []const []u8, offset: u64) std.Io.File.ReadPositionalError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileReadPositional(self.inner.userdata, file, data, offset);
    }

    pub fn fileSeekBy(userdata: ?*anyopaque, file: std.Io.File, relative_offset: i64) std.Io.File.SeekError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileSeekBy(self.inner.userdata, file, relative_offset);
    }

    pub fn fileSeekTo(userdata: ?*anyopaque, file: std.Io.File, absolute_offset: u64) std.Io.File.SeekError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileSeekTo(self.inner.userdata, file, absolute_offset);
    }

    pub fn fileSync(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.SyncError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileSync(self.inner.userdata, file);
    }

    pub fn fileIsTty(userdata: ?*anyopaque, file: std.Io.File) std.Io.Cancelable!bool {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileIsTty(self.inner.userdata, file);
    }

    pub fn fileEnableAnsiEscapeCodes(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.EnableAnsiEscapeCodesError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileEnableAnsiEscapeCodes(self.inner.userdata, file);
    }

    pub fn fileSupportsAnsiEscapeCodes(userdata: ?*anyopaque, file: std.Io.File) std.Io.Cancelable!bool {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileSupportsAnsiEscapeCodes(self.inner.userdata, file);
    }

    pub fn fileSetLength(userdata: ?*anyopaque, file: std.Io.File, length: u64) std.Io.File.SetLengthError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileSetLength(self.inner.userdata, file, length);
    }

    pub fn fileSetOwner(userdata: ?*anyopaque, file: std.Io.File, uid: ?std.Io.File.Uid, gid: ?std.Io.File.Gid) std.Io.File.SetOwnerError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileSetOwner(self.inner.userdata, file, uid, gid);
    }

    pub fn fileSetPermissions(userdata: ?*anyopaque, file: std.Io.File, permissions: std.Io.File.Permissions) std.Io.File.SetPermissionsError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileSetPermissions(self.inner.userdata, file, permissions);
    }

    pub fn fileSetTimestamps(userdata: ?*anyopaque, file: std.Io.File, options: std.Io.File.SetTimestampsOptions) std.Io.File.SetTimestampsError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileSetTimestamps(self.inner.userdata, file, options);
    }

    pub fn fileLock(userdata: ?*anyopaque, file: std.Io.File, lock: std.Io.File.Lock) std.Io.File.LockError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileLock(self.inner.userdata, file, lock);
    }

    pub fn fileTryLock(userdata: ?*anyopaque, file: std.Io.File, lock: std.Io.File.Lock) std.Io.File.LockError!bool {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileTryLock(self.inner.userdata, file, lock);
    }

    pub fn fileUnlock(userdata: ?*anyopaque, file: std.Io.File) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileUnlock(self.inner.userdata, file);
    }

    pub fn fileDowngradeLock(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.DowngradeLockError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileDowngradeLock(self.inner.userdata, file);
    }

    pub fn fileRealPath(userdata: ?*anyopaque, file: std.Io.File, out_buffer: []u8) std.Io.File.RealPathError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileRealPath(self.inner.userdata, file, out_buffer);
    }

    pub fn processExecutableOpen(userdata: ?*anyopaque, flags: std.Io.File.OpenFlags) std.process.OpenExecutableError!std.Io.File {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.processExecutableOpen(self.inner.userdata, flags);
    }

    pub fn processExecutablePath(userdata: ?*anyopaque, buffer: []u8) std.process.ExecutablePathError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.processExecutablePath(self.inner.userdata, buffer);
    }

    pub fn lockStderr(userdata: ?*anyopaque, mode: ?std.Io.Terminal.Mode) std.Io.Cancelable!std.Io.LockedStderr {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.lockStderr(self.inner.userdata, mode);
    }

    pub fn tryLockStderr(userdata: ?*anyopaque, mode: ?std.Io.Terminal.Mode) std.Io.Cancelable!?std.Io.LockedStderr {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.tryLockStderr(self.inner.userdata, mode);
    }

    pub fn unlockStderr(userdata: ?*anyopaque) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.unlockStderr(self.inner.userdata);
    }

    pub fn processSetCurrentDir(userdata: ?*anyopaque, dir: std.Io.Dir) std.process.SetCurrentDirError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.processSetCurrentDir(self.inner.userdata, dir);
    }

    pub fn now(userdata: ?*anyopaque, clock: std.Io.Clock) std.Io.Clock.Error!std.Io.Timestamp {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.now(self.inner.userdata, clock);
    }

    pub fn sleep(userdata: ?*anyopaque, timeout: std.Io.Timeout) std.Io.SleepError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.sleep(self.inner.userdata, timeout);
    }

    pub fn netListenIp(userdata: ?*anyopaque, address: std.Io.net.IpAddress, options: std.Io.net.IpAddress.ListenOptions) std.Io.net.IpAddress.ListenError!std.Io.net.Server {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netListenIp(self.inner.userdata, address, options);
    }

    pub fn netAccept(userdata: ?*anyopaque, server: std.Io.net.Socket.Handle) std.Io.net.Server.AcceptError!std.Io.net.Stream {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netAccept(self.inner.userdata, server);
    }

    pub fn netBindIp(userdata: ?*anyopaque, address: *const std.Io.net.IpAddress, options: std.Io.net.IpAddress.BindOptions) std.Io.net.IpAddress.BindError!std.Io.net.Socket {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netBindIp(self.inner.userdata, address, options);
    }

    pub fn netConnectIp(userdata: ?*anyopaque, address: *const std.Io.net.IpAddress, options: std.Io.net.IpAddress.ConnectOptions) std.Io.net.IpAddress.ConnectError!std.Io.net.Stream {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netConnectIp(self.inner.userdata, address, options);
    }

    pub fn netListenUnix(userdata: ?*anyopaque, address: *const std.Io.net.UnixAddress, options: std.Io.net.UnixAddress.ListenOptions) std.Io.net.UnixAddress.ListenError!std.Io.net.Socket.Handle {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netListenUnix(self.inner.userdata, address, options);
    }

    pub fn netConnectUnix(userdata: ?*anyopaque, address: *const std.Io.net.UnixAddress) std.Io.net.UnixAddress.ConnectError!std.Io.net.Socket.Handle {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netConnectUnix(self.inner.userdata, address);
    }

    pub fn netSend(userdata: ?*anyopaque, handle: std.Io.net.Socket.Handle, msgs: []std.Io.net.OutgoingMessage, flags: std.Io.net.SendFlags) struct { ?std.Io.net.Socket.SendError, usize } {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netSend(self.inner.userdata, handle, msgs, flags);
    }

    pub fn netReceive(userdata: ?*anyopaque, handle: std.Io.net.Socket.Handle, message_buffer: []std.Io.net.IncomingMessage, data_buffer: []u8, flags: std.Io.net.ReceiveFlags, timeout: std.Io.Timeout) struct { ?std.Io.net.Socket.ReceiveTimeoutError, usize } {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netReceive(self.inner.userdata, handle, message_buffer, data_buffer, flags, timeout);
    }

    pub fn netRead(userdata: ?*anyopaque, src: std.Io.net.Socket.Handle, data: [][]u8) std.Io.net.Stream.Reader.Error!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netRead(self.inner.userdata, src, data);
    }

    pub fn netWrite(userdata: ?*anyopaque, dest: std.Io.net.Socket.Handle, header: []const u8, data: []const []const u8, splat: usize) std.Io.net.Stream.Writer.Error!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netWrite(self.inner.userdata, dest, header, data, splat);
    }

    pub fn netWriteFile(userdata: ?*anyopaque, handle: std.Io.net.Socket.Handle, header: []const u8, reader: *std.Io.File.Reader, limit: std.Io.Limit) std.Io.net.Stream.Writer.WriteFileError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netWriteFile(self.inner.userdata, handle, header, reader, limit);
    }

    pub fn netClose(userdata: ?*anyopaque, handles: []const std.Io.net.Socket.Handle) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        self.inner.vtable.netClose(self.inner.userdata, handles);
    }

    pub fn netShutdown(userdata: ?*anyopaque, handle: std.Io.net.Socket.Handle, how: std.Io.net.ShutdownHow) std.Io.net.ShutdownError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netShutdown(self.inner.userdata, handle, how);
    }

    pub fn netInterfaceNameResolve(userdata: ?*anyopaque, name: *const std.Io.net.Interface.Name) std.Io.net.Interface.Name.ResolveError!std.Io.net.Interface {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netInterfaceNameResolve(self.inner.userdata, name);
    }

    pub fn netInterfaceName(userdata: ?*anyopaque, iface: std.Io.net.Interface) std.Io.net.Interface.NameError!std.Io.net.Interface.Name {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netInterfaceName(self.inner.userdata, iface);
    }

    pub fn netLookup(userdata: ?*anyopaque, host: std.Io.net.HostName, q: *std.Io.Queue(std.Io.net.HostName.LookupResult), options: std.Io.net.HostName.LookupOptions) std.Io.net.HostName.LookupError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netLookup(self.inner.userdata, host, q, options);
    }
};
