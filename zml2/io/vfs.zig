const std = @import("std");

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

    pub const HTTP = @import("vfs").HTTP;

    const FsFile = struct {
        backend: std.Io,
        handle: std.Io.File.Handle,
    };

    const FsDir = struct {
        backend: std.Io,
        handle: std.Io.Dir.Handle,
    };

    const ROOT_DIR_HANDLE: std.Io.Dir.Handle = 0;

    allocator: std.mem.Allocator,
    mutex: std.Thread.Mutex,

    backends: std.StringHashMapUnmanaged(std.Io),
    // Maps virtual file and dir handles to backend handles
    files: std.AutoHashMapUnmanaged(std.Io.File.Handle, *FsFile),
    dirs: std.AutoHashMapUnmanaged(std.Io.Dir.Handle, *FsDir),
    next_file_handle: std.Io.File.Handle,
    next_dir_handle: std.Io.Dir.Handle,

    vtable: std.Io.VTable,

    pub fn init(allocator: std.mem.Allocator, base_io: std.Io) VFS {
        var vtable = base_io.vtable.*;
        vtable.dirMake = dirMake;
        vtable.dirMakePath = dirMakePath;
        vtable.dirMakeOpenPath = dirMakeOpenPath;
        vtable.dirStat = dirStat;
        vtable.dirStatPath = dirStatPath;
        vtable.dirAccess = dirAccess;
        vtable.dirCreateFile = dirCreateFile;
        vtable.dirOpenFile = dirOpenFile;
        vtable.dirOpenDir = dirOpenDir;
        vtable.dirClose = dirClose;
        vtable.fileStat = fileStat;
        vtable.fileClose = fileClose;
        vtable.fileWriteStreaming = fileWriteStreaming;
        vtable.fileWritePositional = fileWritePositional;
        vtable.fileReadStreaming = fileReadStreaming;
        vtable.fileReadPositional = fileReadPositional;
        vtable.fileSeekBy = fileSeekBy;
        vtable.fileSeekTo = fileSeekTo;
        vtable.openSelfExe = openSelfExe;

        return .{
            .allocator = allocator,
            .mutex = .{},
            .backends = .{},
            .files = .{},
            .dirs = .{},
            .next_file_handle = 0,
            .next_dir_handle = 1, // 0 reserved for ROOT_DIR_HANDLE
            .vtable = vtable,
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
        self.mutex.lock();
        defer self.mutex.unlock();

        try self.backends.put(self.allocator, scheme, backend);
    }

    pub fn unregister(self: *VFS, scheme: []const u8) bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        return self.backends.remove(scheme);
    }

    pub fn io(self: *VFS) std.Io {
        return .{
            .vtable = &self.vtable,
            .userdata = self,
        };
    }

    fn backendFromAbsolutePath(self: *VFS, path: []const u8) ResolveAbsolutePathError!std.Io {
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

        self.mutex.lock();
        defer self.mutex.unlock();

        const virtual_handle = self.next_file_handle;
        self.next_file_handle += 1;

        try self.files.put(self.allocator, virtual_handle, entry);

        return virtual_handle;
    }

    fn closeFile(self: *VFS, virtual_handle: std.Io.File.Handle) void {
        self.mutex.lock();
        const entry = self.files.fetchRemove(virtual_handle);
        self.mutex.unlock();

        if (entry) |kv| {
            const fs_file = kv.value;
            fs_file.backend.vtable.fileClose(fs_file.backend.userdata, .{ .handle = fs_file.handle });
            self.allocator.destroy(fs_file);
        }
    }

    fn fsFile(self: *VFS, virtual_handle: std.Io.File.Handle) FileEntryError!FsFile {
        self.mutex.lock();
        defer self.mutex.unlock();

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

        self.mutex.lock();
        defer self.mutex.unlock();

        const virtual_handle = self.next_dir_handle;
        self.next_dir_handle += 1;

        try self.dirs.put(self.allocator, virtual_handle, entry);
        return virtual_handle;
    }

    fn closeDir(self: *VFS, virtual_handle: std.Io.Dir.Handle) void {
        if (virtual_handle == ROOT_DIR_HANDLE) return;

        self.mutex.lock();
        const entry = self.dirs.fetchRemove(virtual_handle);
        self.mutex.unlock();

        if (entry) |kv| {
            const fs_dir = kv.value;
            fs_dir.backend.vtable.dirClose(fs_dir.backend.userdata, .{ .handle = fs_dir.handle });
            self.allocator.destroy(fs_dir);
        }
    }

    fn fsDir(self: *VFS, virtual_handle: std.Io.Dir.Handle, sub_path: ?[]const u8) DirEntryError!FsDir {
        if (virtual_handle == ROOT_DIR_HANDLE) {
            std.debug.assert(sub_path != null);
            const backend = try self.backendFromAbsolutePath(sub_path.?);
            return .{ .backend = backend, .handle = ROOT_DIR_HANDLE };
        }

        self.mutex.lock();
        defer self.mutex.unlock();

        const entry = self.dirs.get(virtual_handle) orelse return DirEntryError.MissingVirtualDirHandle;
        return entry.*;
    }

    pub fn rootDir(self: *VFS) std.Io.Dir {
        _ = self;
        return .{ .handle = ROOT_DIR_HANDLE };
    }

    pub fn openAbsoluteDir(self: *VFS, vfs_io: std.Io, path: []const u8, options: std.Io.Dir.OpenOptions) std.Io.Dir.OpenError!std.Io.Dir {
        const backend_dir = try vfs_io.vtable.dirOpenDir(vfs_io.userdata, .{ .handle = ROOT_DIR_HANDLE }, path, options);
        const virtual_handle = self.registerDir(vfs_io, backend_dir.handle) catch return std.Io.Dir.OpenError.Unexpected;

        return .{ .handle = virtual_handle };
    }

    pub fn openAbsoluteFile(self: *VFS, vfs_io: std.Io, path: []const u8, flags: std.Io.File.OpenFlags) std.Io.File.OpenError!std.Io.File {
        const file = try vfs_io.vtable.dirOpenFile(vfs_io.userdata, .{ .handle = ROOT_DIR_HANDLE }, path, flags);
        const virtual_handle = self.registerFile(vfs_io, file.handle) catch return std.Io.File.OpenError.SystemResources;
        return .{ .handle = virtual_handle };
    }

    // VTable implementations

    fn dirMake(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
        sub_path: []const u8,
        mode: std.Io.Dir.Mode,
    ) std.Io.Dir.MakeError!void {
        const self: *VFS = @ptrCast(@alignCast(userdata orelse unreachable));

        const fs_dir = self.fsDir(dir.handle, sub_path) catch |err| {
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

        const fs_dir = self.fsDir(dir.handle, sub_path) catch |err| {
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

        const fs_dir = self.fsDir(dir.handle, sub_path) catch |err| {
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

        if (dir.handle == ROOT_DIR_HANDLE) return std.Io.Dir.StatError.Unexpected;

        const fs_dir = self.fsDir(dir.handle, null) catch |err| {
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

        const fs_dir = self.fsDir(dir.handle, sub_path) catch |err| {
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

        const fs_dir = self.fsDir(dir.handle, sub_path) catch |err| {
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

        const fs_dir = self.fsDir(dir.handle, sub_path) catch |err| {
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

        const fs_dir = self.fsDir(dir.handle, sub_path) catch |err| {
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

        const fs_dir = self.fsDir(dir.handle, sub_path) catch |err| {
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
};
