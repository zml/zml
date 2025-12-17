const std = @import("std");
const vfs = @import("vfs");

const log = std.log.scoped(.@"zml/io/vfs");

pub const VFS = struct {
    pub const Error = FileEntryError || DirEntryError || ResolveAbsolutePathError;

    pub const FileEntryError = error{
        MissingVirtualFileHandle,
    } || ResolveAbsolutePathError;

    pub const DirEntryError = error{
        MissingVirtualDirHandle,
    } || ResolveAbsolutePathError;

    pub const ResolveAbsolutePathError = error{
        SchemeNotRegistered,
        InvalidURI,
    };

    pub const File = vfs.File;
    pub const HTTP = vfs.HTTP;
    pub const HF = vfs.HF;

    const FsFile = struct {
        backend: std.Io,
        handle: std.Io.File.Handle,
    };

    const FsDir = struct {
        backend: std.Io,
        handle: std.Io.Dir.Handle,
    };

    allocator: std.mem.Allocator,
    mutex: std.Io.Mutex,

    backends: std.StringHashMapUnmanaged(std.Io),
    // Maps virtual file and dir handles to backend handles
    files: std.AutoHashMapUnmanaged(std.Io.File.Handle, *FsFile),
    dirs: std.AutoHashMapUnmanaged(std.Io.Dir.Handle, *FsDir),
    next_file_handle: std.Io.File.Handle,
    next_dir_handle: std.Io.Dir.Handle,

    vtable: std.Io.VTable,
    base_io: std.Io,

    pub fn init(allocator: std.mem.Allocator, base_io: std.Io) VFS {
        const vtable: std.Io.VTable = .{
            .async = async,
            .concurrent = concurrent,
            .await = await,
            .cancel = cancel,
            .cancelRequested = cancelRequested,
            .groupAsync = groupAsync,
            .groupConcurrent = groupConcurrent,
            .groupWait = groupWait,
            .groupCancel = groupCancel,
            .select = select,
            .mutexLock = mutexLock,
            .mutexLockUncancelable = mutexLockUncancelable,
            .mutexUnlock = mutexUnlock,
            .conditionWait = conditionWait,
            .conditionWaitUncancelable = conditionWaitUncancelable,
            .conditionWake = conditionWake,
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
            .netClose = netClose,
            .netInterfaceNameResolve = netInterfaceNameResolve,
            .netInterfaceName = netInterfaceName,
            .netLookup = netLookup,
            .dirMake = dirMake,
            .dirMakePath = dirMakePath,
            .dirMakeOpenPath = dirMakeOpenPath,
            .dirStat = dirStat,
            .dirStatPath = dirStatPath,
            .dirAccess = dirAccess,
            .dirCreateFile = dirCreateFile,
            .dirOpenFile = dirOpenFile,
            .dirOpenDir = dirOpenDir,
            .dirClose = dirClose,
            .fileStat = fileStat,
            .fileClose = fileClose,
            .fileWriteStreaming = fileWriteStreaming,
            .fileWritePositional = fileWritePositional,
            .fileReadStreaming = fileReadStreaming,
            .fileReadPositional = fileReadPositional,
            .fileSeekBy = fileSeekBy,
            .fileSeekTo = fileSeekTo,
            .openSelfExe = openSelfExe,
        };

        return .{
            .allocator = allocator,
            .mutex = .init,
            .backends = .{},
            .files = .{},
            .dirs = .{},
            .next_file_handle = 0,
            .next_dir_handle = 0,
            .vtable = vtable,
            .base_io = base_io,
        };
    }

    pub fn deinit(self: *VFS) void {
        var file_it = self.files.valueIterator();
        while (file_it.next()) |entry_ptr| {
            self.allocator.destroy(entry_ptr.*);
        }
        self.files.deinit(self.allocator);

        var dir_it = self.dirs.valueIterator();
        while (dir_it.next()) |entry_ptr| {
            self.allocator.destroy(entry_ptr.*);
        }
        self.dirs.deinit(self.allocator);

        self.backends.deinit(self.allocator);
    }

    pub fn register(self: *VFS, scheme: []const u8, backend: std.Io) std.mem.Allocator.Error!void {
        self.mutex.lockUncancelable(self.base_io);
        defer self.mutex.unlock(self.base_io);

        try self.backends.put(self.allocator, scheme, backend);
    }

    pub fn unregister(self: *VFS, scheme: []const u8) bool {
        self.mutex.lockUncancelable(self.base_io);
        defer self.mutex.unlock(self.base_io);

        return self.backends.remove(scheme);
    }

    pub fn io(self: *VFS) std.Io {
        return .{
            .vtable = &self.vtable,
            .userdata = self,
        };
    }

    fn backendFromAbsolutePath(self: *VFS, path: []const u8) ResolveAbsolutePathError!std.Io {
        if (std.fs.path.isAbsolutePosix(path)) return self.base_io;
        const uri = std.Uri.parse(path) catch return ResolveAbsolutePathError.InvalidURI;
        return self.backends.get(uri.scheme) orelse return ResolveAbsolutePathError.SchemeNotRegistered;
    }

    fn registerFile(self: *VFS, backend: std.Io, handle: std.Io.File.Handle) std.mem.Allocator.Error!std.Io.File.Handle {
        const entry = try self.allocator.create(FsFile);
        errdefer self.allocator.destroy(entry);

        entry.* = .{
            .backend = backend,
            .handle = handle,
        };

        self.mutex.lockUncancelable(self.base_io);
        defer self.mutex.unlock(self.base_io);

        const virtual_handle = self.next_file_handle;
        self.next_file_handle += 1;

        try self.files.put(self.allocator, virtual_handle, entry);

        return virtual_handle;
    }

    fn closeFile(self: *VFS, virtual_handle: std.Io.File.Handle) void {
        self.mutex.lockUncancelable(self.base_io);
        const entry = self.files.fetchRemove(virtual_handle);
        self.mutex.unlock(self.base_io);

        if (entry) |kv| {
            const fs_file = kv.value;
            fs_file.backend.vtable.fileClose(fs_file.backend.userdata, .{ .handle = fs_file.handle });
            self.allocator.destroy(fs_file);
        }
    }

    fn fsFile(self: *VFS, virtual_handle: std.Io.File.Handle) FileEntryError!FsFile {
        self.mutex.lockUncancelable(self.base_io);
        defer self.mutex.unlock(self.base_io);

        const entry = self.files.get(virtual_handle) orelse return FileEntryError.MissingVirtualFileHandle;
        return entry.*;
    }

    fn registerDir(self: *VFS, backend: std.Io, handle: std.Io.Dir.Handle) std.mem.Allocator.Error!std.Io.Dir.Handle {
        const entry = try self.allocator.create(FsDir);
        errdefer self.allocator.destroy(entry);

        entry.* = .{
            .backend = backend,
            .handle = handle,
        };

        self.mutex.lockUncancelable(self.base_io);
        defer self.mutex.unlock(self.base_io);

        const virtual_handle = self.next_dir_handle;
        self.next_dir_handle += 1;

        try self.dirs.put(self.allocator, virtual_handle, entry);
        return virtual_handle;
    }

    fn closeDir(self: *VFS, virtual_handle: std.Io.Dir.Handle) void {
        self.mutex.lockUncancelable(self.base_io);
        const entry = self.dirs.fetchRemove(virtual_handle);
        self.mutex.unlock(self.base_io);

        if (entry) |kv| {
            const fs_dir = kv.value;
            fs_dir.backend.vtable.dirClose(fs_dir.backend.userdata, .{ .handle = fs_dir.handle });
            self.allocator.destroy(fs_dir);
        }
    }

    fn fsDir(self: *VFS, dir: std.Io.Dir, sub_path: ?[]const u8) DirEntryError!FsDir {
        if (std.meta.eql(dir, std.Io.Dir.cwd())) {
            std.debug.assert(sub_path != null);
            const backend = self.backendFromAbsolutePath(sub_path.?) catch self.base_io;
            return .{ .backend = backend, .handle = dir.handle };
        }

        self.mutex.lockUncancelable(self.base_io);
        defer self.mutex.unlock(self.base_io);

        const entry = self.dirs.get(dir.handle) orelse return DirEntryError.MissingVirtualDirHandle;
        return entry.*;
    }

    // VTable implementations

    fn dirMake(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
        sub_path: []const u8,
        mode: std.Io.Dir.Mode,
    ) std.Io.Dir.MakeError!void {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));

        const fs_dir = self.fsDir(dir, sub_path) catch |err| {
            return switch (err) {
                DirEntryError.SchemeNotRegistered => std.Io.Dir.MakeError.BadPathName,
                DirEntryError.InvalidURI => std.Io.Dir.MakeError.BadPathName,
                DirEntryError.MissingVirtualDirHandle => std.Io.Dir.MakeError.FileNotFound,
            };
        };

        return fs_dir.backend.vtable.dirMake(fs_dir.backend.userdata, .{ .handle = fs_dir.handle }, sub_path, mode);
    }

    fn dirMakePath(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
        sub_path: []const u8,
        mode: std.Io.Dir.Mode,
    ) std.Io.Dir.MakeError!void {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));

        const fs_dir = self.fsDir(dir, sub_path) catch |err| {
            return switch (err) {
                DirEntryError.SchemeNotRegistered => std.Io.Dir.MakeError.BadPathName,
                DirEntryError.InvalidURI => std.Io.Dir.MakeError.BadPathName,
                DirEntryError.MissingVirtualDirHandle => std.Io.Dir.MakeError.FileNotFound,
            };
        };

        return fs_dir.backend.vtable.dirMakePath(fs_dir.backend.userdata, .{ .handle = fs_dir.handle }, sub_path, mode);
    }

    fn dirMakeOpenPath(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
        sub_path: []const u8,
        options: std.Io.Dir.OpenOptions,
    ) std.Io.Dir.MakeOpenPathError!std.Io.Dir {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));

        const fs_dir = self.fsDir(dir, sub_path) catch |err| {
            return switch (err) {
                DirEntryError.SchemeNotRegistered => std.Io.Dir.MakeOpenPathError.BadPathName,
                DirEntryError.InvalidURI => std.Io.Dir.MakeOpenPathError.BadPathName,
                DirEntryError.MissingVirtualDirHandle => std.Io.Dir.MakeOpenPathError.FileNotFound,
            };
        };

        const fs_open_dir = try fs_dir.backend.vtable.dirMakeOpenPath(fs_dir.backend.userdata, .{ .handle = fs_dir.handle }, sub_path, options);

        const vfs_handle = self.registerDir(fs_dir.backend, fs_open_dir.handle) catch |err| {
            return switch (err) {
                std.mem.Allocator.Error.OutOfMemory => std.Io.Dir.OpenError.SystemFdQuotaExceeded,
            };
        };

        return .{ .handle = vfs_handle };
    }

    fn dirStat(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
    ) std.Io.Dir.StatError!std.Io.Dir.Stat {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));

        const fs_dir = self.fsDir(dir, null) catch |err| {
            return switch (err) {
                DirEntryError.SchemeNotRegistered => std.Io.Dir.StatError.Unexpected,
                DirEntryError.InvalidURI => std.Io.Dir.StatError.Unexpected,
                DirEntryError.MissingVirtualDirHandle => std.Io.Dir.StatError.Unexpected,
            };
        };

        return fs_dir.backend.vtable.dirStat(fs_dir.backend.userdata, .{ .handle = fs_dir.handle });
    }

    fn dirStatPath(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
        sub_path: []const u8,
        options: std.Io.Dir.StatPathOptions,
    ) std.Io.Dir.StatPathError!std.Io.Dir.Stat {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));

        const fs_dir = self.fsDir(dir, sub_path) catch |err| {
            return switch (err) {
                DirEntryError.SchemeNotRegistered => std.Io.Dir.StatPathError.BadPathName,
                DirEntryError.InvalidURI => std.Io.Dir.StatPathError.BadPathName,
                DirEntryError.MissingVirtualDirHandle => std.Io.Dir.StatPathError.FileNotFound,
            };
        };

        return fs_dir.backend.vtable.dirStatPath(fs_dir.backend.userdata, .{ .handle = fs_dir.handle }, sub_path, options);
    }

    fn dirAccess(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
        sub_path: []const u8,
        options: std.Io.Dir.AccessOptions,
    ) std.Io.Dir.AccessError!void {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));

        const fs_dir = self.fsDir(dir, sub_path) catch |err| {
            return switch (err) {
                DirEntryError.SchemeNotRegistered => std.Io.Dir.AccessError.BadPathName,
                DirEntryError.InvalidURI => std.Io.Dir.AccessError.BadPathName,
                DirEntryError.MissingVirtualDirHandle => std.Io.Dir.AccessError.FileNotFound,
            };
        };

        return fs_dir.backend.vtable.dirAccess(fs_dir.backend.userdata, .{ .handle = fs_dir.handle }, sub_path, options);
    }

    fn dirCreateFile(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
        sub_path: []const u8,
        flags: std.Io.File.CreateFlags,
    ) std.Io.File.OpenError!std.Io.File {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));

        const fs_dir = self.fsDir(dir, sub_path) catch |err| {
            return switch (err) {
                DirEntryError.SchemeNotRegistered => std.Io.File.OpenError.BadPathName,
                DirEntryError.InvalidURI => std.Io.File.OpenError.BadPathName,
                DirEntryError.MissingVirtualDirHandle => std.Io.File.OpenError.FileNotFound,
            };
        };

        const fs_file = try fs_dir.backend.vtable.dirCreateFile(fs_dir.backend.userdata, .{ .handle = fs_dir.handle }, sub_path, flags);

        const vfs_handle = self.registerFile(fs_dir.backend, fs_file.handle) catch |err| {
            return switch (err) {
                std.mem.Allocator.Error.OutOfMemory => std.Io.File.OpenError.SystemFdQuotaExceeded,
            };
        };

        return .{ .handle = vfs_handle };
    }

    fn dirOpenFile(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
        sub_path: []const u8,
        flags: std.Io.File.OpenFlags,
    ) std.Io.File.OpenError!std.Io.File {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));

        const fs_dir = self.fsDir(dir, sub_path) catch |err| {
            return switch (err) {
                DirEntryError.SchemeNotRegistered => std.Io.File.OpenError.BadPathName,
                DirEntryError.InvalidURI => std.Io.File.OpenError.BadPathName,
                DirEntryError.MissingVirtualDirHandle => std.Io.File.OpenError.FileNotFound,
            };
        };

        const fs_file = try fs_dir.backend.vtable.dirOpenFile(fs_dir.backend.userdata, .{ .handle = fs_dir.handle }, sub_path, flags);

        const vfs_handle = self.registerFile(fs_dir.backend, fs_file.handle) catch |err| {
            return switch (err) {
                std.mem.Allocator.Error.OutOfMemory => std.Io.File.OpenError.SystemFdQuotaExceeded,
            };
        };

        return .{ .handle = vfs_handle };
    }

    fn dirOpenDir(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
        sub_path: []const u8,
        options: std.Io.Dir.OpenOptions,
    ) std.Io.Dir.OpenError!std.Io.Dir {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));

        const fs_dir = self.fsDir(dir, sub_path) catch |err| {
            return switch (err) {
                DirEntryError.SchemeNotRegistered => std.Io.Dir.OpenError.BadPathName,
                DirEntryError.InvalidURI => std.Io.Dir.OpenError.BadPathName,
                DirEntryError.MissingVirtualDirHandle => std.Io.Dir.OpenError.FileNotFound,
            };
        };

        const fs_sub_dir = try fs_dir.backend.vtable.dirOpenDir(fs_dir.backend.userdata, .{ .handle = fs_dir.handle }, sub_path, options);

        const vfs_handle = self.registerDir(fs_dir.backend, fs_sub_dir.handle) catch |err| {
            return switch (err) {
                std.mem.Allocator.Error.OutOfMemory => std.Io.Dir.OpenError.SystemFdQuotaExceeded,
            };
        };

        return .{ .handle = vfs_handle };
    }

    fn dirClose(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
    ) void {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));
        self.closeDir(dir.handle);
    }

    fn fileStat(
        userdata: ?*anyopaque,
        file: std.Io.File,
    ) std.Io.File.StatError!std.Io.File.Stat {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));

        const fs_file = self.fsFile(file.handle) catch |err| {
            return switch (err) {
                FileEntryError.SchemeNotRegistered => std.Io.File.StatError.Unexpected,
                FileEntryError.InvalidURI => std.Io.File.StatError.Unexpected,
                FileEntryError.MissingVirtualFileHandle => std.Io.File.StatError.Unexpected,
            };
        };

        return fs_file.backend.vtable.fileStat(fs_file.backend.userdata, .{ .handle = fs_file.handle });
    }

    fn fileClose(
        userdata: ?*anyopaque,
        file: std.Io.File,
    ) void {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));
        self.closeFile(file.handle);
    }

    fn fileWriteStreaming(
        userdata: ?*anyopaque,
        file: std.Io.File,
        buffer: [][]const u8,
    ) std.Io.File.WriteStreamingError!usize {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));

        const fs_file = self.fsFile(file.handle) catch |err| {
            return switch (err) {
                FileEntryError.SchemeNotRegistered => std.Io.File.WriteStreamingError.Unexpected,
                FileEntryError.InvalidURI => std.Io.File.WriteStreamingError.Unexpected,
                FileEntryError.MissingVirtualFileHandle => std.Io.File.WriteStreamingError.Unexpected,
            };
        };

        return fs_file.backend.vtable.fileWriteStreaming(fs_file.backend.userdata, .{ .handle = fs_file.handle }, buffer);
    }

    fn fileWritePositional(
        userdata: ?*anyopaque,
        file: std.Io.File,
        buffer: [][]const u8,
        offset: u64,
    ) std.Io.File.WritePositionalError!usize {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));

        const fs_file = self.fsFile(file.handle) catch |err| {
            return switch (err) {
                FileEntryError.SchemeNotRegistered => std.Io.File.WritePositionalError.Unexpected,
                FileEntryError.InvalidURI => std.Io.File.WritePositionalError.Unexpected,
                FileEntryError.MissingVirtualFileHandle => std.Io.File.WritePositionalError.Unexpected,
            };
        };

        return fs_file.backend.vtable.fileWritePositional(fs_file.backend.userdata, .{ .handle = fs_file.handle }, buffer, offset);
    }

    fn fileReadStreaming(
        userdata: ?*anyopaque,
        file: std.Io.File,
        data: [][]u8,
    ) std.Io.File.Reader.Error!usize {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));

        const fs_file = self.fsFile(file.handle) catch |err| {
            return switch (err) {
                FileEntryError.SchemeNotRegistered => std.Io.File.Reader.Error.Unexpected,
                FileEntryError.InvalidURI => std.Io.File.Reader.Error.Unexpected,
                FileEntryError.MissingVirtualFileHandle => std.Io.File.Reader.Error.Unexpected,
            };
        };

        return fs_file.backend.vtable.fileReadStreaming(fs_file.backend.userdata, .{ .handle = fs_file.handle }, data);
    }

    fn fileReadPositional(
        userdata: ?*anyopaque,
        file: std.Io.File,
        data: [][]u8,
        offset: u64,
    ) std.Io.File.ReadPositionalError!usize {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));

        const fs_file = self.fsFile(file.handle) catch |err| {
            return switch (err) {
                FileEntryError.SchemeNotRegistered => std.Io.File.ReadPositionalError.Unexpected,
                FileEntryError.InvalidURI => std.Io.File.ReadPositionalError.Unexpected,
                FileEntryError.MissingVirtualFileHandle => std.Io.File.ReadPositionalError.Unexpected,
            };
        };

        return fs_file.backend.vtable.fileReadPositional(fs_file.backend.userdata, .{ .handle = fs_file.handle }, data, offset);
    }

    fn fileSeekBy(
        userdata: ?*anyopaque,
        file: std.Io.File,
        relative_offset: i64,
    ) std.Io.File.SeekError!void {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));

        const fs_file = self.fsFile(file.handle) catch |err| {
            return switch (err) {
                FileEntryError.SchemeNotRegistered => std.Io.File.SeekError.Unexpected,
                FileEntryError.InvalidURI => std.Io.File.SeekError.Unexpected,
                FileEntryError.MissingVirtualFileHandle => std.Io.File.SeekError.Unexpected,
            };
        };

        return fs_file.backend.vtable.fileSeekBy(fs_file.backend.userdata, .{ .handle = fs_file.handle }, relative_offset);
    }

    fn fileSeekTo(
        userdata: ?*anyopaque,
        file: std.Io.File,
        absolute_offset: u64,
    ) std.Io.File.SeekError!void {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));

        const fs_file = self.fsFile(file.handle) catch |err| {
            return switch (err) {
                FileEntryError.SchemeNotRegistered => std.Io.File.SeekError.Unexpected,
                FileEntryError.InvalidURI => std.Io.File.SeekError.Unexpected,
                FileEntryError.MissingVirtualFileHandle => std.Io.File.SeekError.Unexpected,
            };
        };

        return fs_file.backend.vtable.fileSeekTo(fs_file.backend.userdata, .{ .handle = fs_file.handle }, absolute_offset);
    }

    fn openSelfExe(userdata: ?*anyopaque, flags: std.Io.File.OpenFlags) std.Io.File.OpenSelfExeError!std.Io.File {
        _ = userdata;
        _ = flags;
        return std.Io.File.OpenSelfExeError.NotSupported;
    }

    // Forwarding wrappers

    fn async(
        userdata: ?*anyopaque,
        result: []u8,
        result_alignment: std.mem.Alignment,
        context: []const u8,
        context_alignment: std.mem.Alignment,
        start: *const fn (context: *const anyopaque, result: *anyopaque) void,
    ) ?*std.Io.AnyFuture {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));
        const f = self.base_io.vtable.async;
        return f(self.base_io.userdata, result, result_alignment, context, context_alignment, start);
    }

    fn concurrent(
        userdata: ?*anyopaque,
        result_len: usize,
        result_alignment: std.mem.Alignment,
        context: []const u8,
        context_alignment: std.mem.Alignment,
        start: *const fn (context: *const anyopaque, result: *anyopaque) void,
    ) std.Io.ConcurrentError!*std.Io.AnyFuture {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));
        const f = self.base_io.vtable.concurrent;
        return f(self.base_io.userdata, result_len, result_alignment, context, context_alignment, start);
    }

    fn await(
        userdata: ?*anyopaque,
        any_future: *std.Io.AnyFuture,
        result: []u8,
        result_alignment: std.mem.Alignment,
    ) void {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));
        const f = self.base_io.vtable.await;
        f(self.base_io.userdata, any_future, result, result_alignment);
    }

    fn cancel(
        userdata: ?*anyopaque,
        any_future: *std.Io.AnyFuture,
        result: []u8,
        result_alignment: std.mem.Alignment,
    ) void {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));
        const f = self.base_io.vtable.cancel;
        f(self.base_io.userdata, any_future, result, result_alignment);
    }

    fn cancelRequested(userdata: ?*anyopaque) bool {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));
        const f = self.base_io.vtable.cancelRequested;
        return f(self.base_io.userdata);
    }

    fn groupAsync(
        userdata: ?*anyopaque,
        group: *std.Io.Group,
        context: []const u8,
        context_alignment: std.mem.Alignment,
        start: *const fn (*std.Io.Group, context: *const anyopaque) void,
    ) void {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));
        const f = self.base_io.vtable.groupAsync;
        f(self.base_io.userdata, group, context, context_alignment, start);
    }

    fn groupConcurrent(
        userdata: ?*anyopaque,
        group: *std.Io.Group,
        context: []const u8,
        context_alignment: std.mem.Alignment,
        start: *const fn (*std.Io.Group, context: *const anyopaque) void,
    ) std.Io.ConcurrentError!void {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.base_io.vtable.groupConcurrent(self.base_io.userdata, group, context, context_alignment, start);
    }

    fn groupWait(userdata: ?*anyopaque, group: *std.Io.Group, token: *anyopaque) void {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));
        const f = self.base_io.vtable.groupWait;
        f(self.base_io.userdata, group, token);
    }

    fn groupCancel(userdata: ?*anyopaque, group: *std.Io.Group, token: *anyopaque) void {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));
        const f = self.base_io.vtable.groupCancel;
        f(self.base_io.userdata, group, token);
    }

    fn select(userdata: ?*anyopaque, futures: []const *std.Io.AnyFuture) std.Io.Cancelable!usize {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.base_io.vtable.select(self.base_io.userdata, futures);
    }

    fn mutexLock(userdata: ?*anyopaque, prev_state: std.Io.Mutex.State, mutex: *std.Io.Mutex) std.Io.Cancelable!void {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.base_io.vtable.mutexLock(self.base_io.userdata, prev_state, mutex);
    }

    fn mutexLockUncancelable(userdata: ?*anyopaque, prev_state: std.Io.Mutex.State, mutex: *std.Io.Mutex) void {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));
        const f = self.base_io.vtable.mutexLockUncancelable;
        f(self.base_io.userdata, prev_state, mutex);
    }

    fn mutexUnlock(userdata: ?*anyopaque, prev_state: std.Io.Mutex.State, mutex: *std.Io.Mutex) void {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));
        const f = self.base_io.vtable.mutexUnlock;
        f(self.base_io.userdata, prev_state, mutex);
    }

    fn conditionWait(userdata: ?*anyopaque, cond: *std.Io.Condition, mutex: *std.Io.Mutex) std.Io.Cancelable!void {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.base_io.vtable.conditionWait(self.base_io.userdata, cond, mutex);
    }

    fn conditionWaitUncancelable(userdata: ?*anyopaque, cond: *std.Io.Condition, mutex: *std.Io.Mutex) void {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));
        const f = self.base_io.vtable.conditionWaitUncancelable;
        f(self.base_io.userdata, cond, mutex);
    }

    fn conditionWake(userdata: ?*anyopaque, cond: *std.Io.Condition, wake: std.Io.Condition.Wake) void {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));
        const f = self.base_io.vtable.conditionWake;
        f(self.base_io.userdata, cond, wake);
    }

    fn now(userdata: ?*anyopaque, clock: std.Io.Clock) std.Io.Clock.Error!std.Io.Timestamp {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.base_io.vtable.now(self.base_io.userdata, clock);
    }

    fn sleep(userdata: ?*anyopaque, timeout: std.Io.Timeout) std.Io.SleepError!void {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.base_io.vtable.sleep(self.base_io.userdata, timeout);
    }

    fn netListenIp(
        userdata: ?*anyopaque,
        address: std.Io.net.IpAddress,
        options: std.Io.net.IpAddress.ListenOptions,
    ) std.Io.net.IpAddress.ListenError!std.Io.net.Server {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.base_io.vtable.netListenIp(self.base_io.userdata, address, options);
    }

    fn netAccept(userdata: ?*anyopaque, server: std.Io.net.Socket.Handle) std.Io.net.Server.AcceptError!std.Io.net.Stream {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.base_io.vtable.netAccept(self.base_io.userdata, server);
    }

    fn netBindIp(
        userdata: ?*anyopaque,
        address: *const std.Io.net.IpAddress,
        options: std.Io.net.IpAddress.BindOptions,
    ) std.Io.net.IpAddress.BindError!std.Io.net.Socket {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.base_io.vtable.netBindIp(self.base_io.userdata, address, options);
    }

    fn netConnectIp(
        userdata: ?*anyopaque,
        address: *const std.Io.net.IpAddress,
        options: std.Io.net.IpAddress.ConnectOptions,
    ) std.Io.net.IpAddress.ConnectError!std.Io.net.Stream {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.base_io.vtable.netConnectIp(self.base_io.userdata, address, options);
    }

    fn netListenUnix(
        userdata: ?*anyopaque,
        address: *const std.Io.net.UnixAddress,
        options: std.Io.net.UnixAddress.ListenOptions,
    ) std.Io.net.UnixAddress.ListenError!std.Io.net.Socket.Handle {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.base_io.vtable.netListenUnix(self.base_io.userdata, address, options);
    }

    fn netConnectUnix(
        userdata: ?*anyopaque,
        address: *const std.Io.net.UnixAddress,
    ) std.Io.net.UnixAddress.ConnectError!std.Io.net.Socket.Handle {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.base_io.vtable.netConnectUnix(self.base_io.userdata, address);
    }

    fn netSend(
        userdata: ?*anyopaque,
        handle: std.Io.net.Socket.Handle,
        msgs: []std.Io.net.OutgoingMessage,
        flags: std.Io.net.SendFlags,
    ) struct { ?std.Io.net.Socket.SendError, usize } {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.base_io.vtable.netSend(self.base_io.userdata, handle, msgs, flags);
    }

    fn netReceive(
        userdata: ?*anyopaque,
        handle: std.Io.net.Socket.Handle,
        message_buffer: []std.Io.net.IncomingMessage,
        data_buffer: []u8,
        flags: std.Io.net.ReceiveFlags,
        timeout: std.Io.Timeout,
    ) struct { ?std.Io.net.Socket.ReceiveTimeoutError, usize } {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.base_io.vtable.netReceive(self.base_io.userdata, handle, message_buffer, data_buffer, flags, timeout);
    }

    fn netRead(userdata: ?*anyopaque, src: std.Io.net.Socket.Handle, data: [][]u8) std.Io.net.Stream.Reader.Error!usize {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.base_io.vtable.netRead(self.base_io.userdata, src, data);
    }

    fn netWrite(
        userdata: ?*anyopaque,
        dest: std.Io.net.Socket.Handle,
        header: []const u8,
        data: []const []const u8,
        splat: usize,
    ) std.Io.net.Stream.Writer.Error!usize {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.base_io.vtable.netWrite(self.base_io.userdata, dest, header, data, splat);
    }

    fn netClose(userdata: ?*anyopaque, handle: std.Io.net.Socket.Handle) void {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));
        const f = self.base_io.vtable.netClose;
        f(self.base_io.userdata, handle);
    }

    fn netInterfaceNameResolve(
        userdata: ?*anyopaque,
        name: *const std.Io.net.Interface.Name,
    ) std.Io.net.Interface.Name.ResolveError!std.Io.net.Interface {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.base_io.vtable.netInterfaceNameResolve(self.base_io.userdata, name);
    }

    fn netInterfaceName(
        userdata: ?*anyopaque,
        iface: std.Io.net.Interface,
    ) std.Io.net.Interface.NameError!std.Io.net.Interface.Name {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.base_io.vtable.netInterfaceName(self.base_io.userdata, iface);
    }

    fn netLookup(
        userdata: ?*anyopaque,
        host: std.Io.net.HostName,
        q: *std.Io.Queue(std.Io.net.HostName.LookupResult),
        options: std.Io.net.HostName.LookupOptions,
    ) void {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));
        const f = self.base_io.vtable.netLookup;
        f(self.base_io.userdata, host, q, options);
    }
};
