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
        };
        for (std.meta.fieldNames(@TypeOf(overrides))) |field_name| {
            @field(new_vtable, field_name) = @field(overrides, field_name);
        }
        return new_vtable;
    }

    pub fn as(userdata: ?*anyopaque) *VFSBase {
        return @ptrCast(@alignCast(userdata orelse unreachable));
    }

    pub fn dirMake(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, mode: std.Io.Dir.Mode) std.Io.Dir.MakeError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.dirMake(self.inner.userdata, dir, sub_path, mode);
    }

    pub fn dirMakePath(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, mode: std.Io.Dir.Mode) std.Io.Dir.MakeError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.dirMakePath(self.inner.userdata, dir, sub_path, mode);
    }

    pub fn dirMakeOpenPath(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, options: std.Io.Dir.OpenOptions) std.Io.Dir.MakeOpenPathError!std.Io.Dir {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.dirMakeOpenPath(self.inner.userdata, dir, sub_path, options);
    }

    pub fn dirStat(userdata: ?*anyopaque, dir: std.Io.Dir) std.Io.Dir.StatError!std.Io.Dir.Stat {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.dirStat(self.inner.userdata, dir);
    }

    pub fn dirStatPath(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, options: std.Io.Dir.StatPathOptions) std.Io.Dir.StatPathError!std.Io.Dir.Stat {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.dirStatPath(self.inner.userdata, dir, sub_path, options);
    }

    pub fn dirAccess(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, options: std.Io.Dir.AccessOptions) std.Io.Dir.AccessError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.dirAccess(self.inner.userdata, dir, sub_path, options);
    }

    pub fn dirCreateFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, flags: std.Io.File.CreateFlags) std.Io.File.OpenError!std.Io.File {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.dirCreateFile(self.inner.userdata, dir, sub_path, flags);
    }

    pub fn dirOpenFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, flags: std.Io.File.OpenFlags) std.Io.File.OpenError!std.Io.File {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.dirOpenFile(self.inner.userdata, dir, sub_path, flags);
    }

    pub fn dirOpenDir(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, options: std.Io.Dir.OpenOptions) std.Io.Dir.OpenError!std.Io.Dir {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.dirOpenDir(self.inner.userdata, dir, sub_path, options);
    }

    pub fn dirClose(userdata: ?*anyopaque, dir: std.Io.Dir) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        self.inner.vtable.dirClose(self.inner.userdata, dir);
    }

    pub fn fileStat(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.StatError!std.Io.File.Stat {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.fileStat(self.inner.userdata, file);
    }

    pub fn fileClose(userdata: ?*anyopaque, file: std.Io.File) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        self.inner.vtable.fileClose(self.inner.userdata, file);
    }

    pub fn fileWriteStreaming(userdata: ?*anyopaque, file: std.Io.File, buffer: [][]const u8) std.Io.File.WriteStreamingError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.fileWriteStreaming(self.inner.userdata, file, buffer);
    }

    pub fn fileWritePositional(userdata: ?*anyopaque, file: std.Io.File, buffer: [][]const u8, offset: u64) std.Io.File.WritePositionalError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.fileWritePositional(self.inner.userdata, file, buffer, offset);
    }

    pub fn fileReadStreaming(userdata: ?*anyopaque, file: std.Io.File, data: [][]u8) std.Io.File.Reader.Error!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.fileReadStreaming(self.inner.userdata, file, data);
    }

    pub fn fileReadPositional(userdata: ?*anyopaque, file: std.Io.File, data: [][]u8, offset: u64) std.Io.File.ReadPositionalError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.fileReadPositional(self.inner.userdata, file, data, offset);
    }

    pub fn fileSeekBy(userdata: ?*anyopaque, file: std.Io.File, relative_offset: i64) std.Io.File.SeekError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.fileSeekBy(self.inner.userdata, file, relative_offset);
    }

    pub fn fileSeekTo(userdata: ?*anyopaque, file: std.Io.File, absolute_offset: u64) std.Io.File.SeekError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.fileSeekTo(self.inner.userdata, file, absolute_offset);
    }

    pub fn openSelfExe(userdata: ?*anyopaque, flags: std.Io.File.OpenFlags) std.Io.File.OpenSelfExeError!std.Io.File {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.openSelfExe(self.inner.userdata, flags);
    }

    pub fn async(userdata: ?*anyopaque, result: []u8, result_alignment: std.mem.Alignment, context: []const u8, context_alignment: std.mem.Alignment, start: *const fn (context: *const anyopaque, result: *anyopaque) void) ?*std.Io.AnyFuture {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.async(self.inner.userdata, result, result_alignment, context, context_alignment, start);
    }

    pub fn concurrent(userdata: ?*anyopaque, result_len: usize, result_alignment: std.mem.Alignment, context: []const u8, context_alignment: std.mem.Alignment, start: *const fn (context: *const anyopaque, result: *anyopaque) void) std.Io.ConcurrentError!*std.Io.AnyFuture {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.concurrent(self.inner.userdata, result_len, result_alignment, context, context_alignment, start);
    }

    pub fn await(userdata: ?*anyopaque, any_future: *std.Io.AnyFuture, result: []u8, result_alignment: std.mem.Alignment) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.await(self.inner.userdata, any_future, result, result_alignment);
    }

    pub fn cancel(userdata: ?*anyopaque, any_future: *std.Io.AnyFuture, result: []u8, result_alignment: std.mem.Alignment) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.cancel(self.inner.userdata, any_future, result, result_alignment);
    }

    pub fn cancelRequested(userdata: ?*anyopaque) bool {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.cancelRequested(self.inner.userdata);
    }

    pub fn groupAsync(userdata: ?*anyopaque, group: *std.Io.Group, context: []const u8, context_alignment: std.mem.Alignment, start: *const fn (*std.Io.Group, context: *const anyopaque) void) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.groupAsync(self.inner.userdata, group, context, context_alignment, start);
    }

    pub fn groupConcurrent(userdata: ?*anyopaque, group: *std.Io.Group, context: []const u8, context_alignment: std.mem.Alignment, start: *const fn (*std.Io.Group, context: *const anyopaque) void) std.Io.ConcurrentError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.groupConcurrent(self.inner.userdata, group, context, context_alignment, start);
    }

    pub fn groupWait(userdata: ?*anyopaque, group: *std.Io.Group, token: *anyopaque) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.groupWait(self.inner.userdata, group, token);
    }

    pub fn groupCancel(userdata: ?*anyopaque, group: *std.Io.Group, token: *anyopaque) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.groupCancel(self.inner.userdata, group, token);
    }

    pub fn select(userdata: ?*anyopaque, futures: []const *std.Io.AnyFuture) std.Io.Cancelable!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.select(self.inner.userdata, futures);
    }

    pub fn mutexLock(userdata: ?*anyopaque, prev_state: std.Io.Mutex.State, mutex: *std.Io.Mutex) std.Io.Cancelable!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.mutexLock(self.inner.userdata, prev_state, mutex);
    }

    pub fn mutexLockUncancelable(userdata: ?*anyopaque, prev_state: std.Io.Mutex.State, mutex: *std.Io.Mutex) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        std.debug.print(">>>>>>>> {*}\n", .{self});
        self.inner.vtable.mutexLockUncancelable(self.inner.userdata, prev_state, mutex);
    }

    pub fn mutexUnlock(userdata: ?*anyopaque, prev_state: std.Io.Mutex.State, mutex: *std.Io.Mutex) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        self.inner.vtable.mutexUnlock(self.inner.userdata, prev_state, mutex);
    }

    pub fn conditionWait(userdata: ?*anyopaque, cond: *std.Io.Condition, mutex: *std.Io.Mutex) std.Io.Cancelable!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.conditionWait(self.inner.userdata, cond, mutex);
    }

    pub fn conditionWaitUncancelable(userdata: ?*anyopaque, cond: *std.Io.Condition, mutex: *std.Io.Mutex) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        self.inner.vtable.conditionWaitUncancelable(self.inner.userdata, cond, mutex);
    }

    pub fn conditionWake(userdata: ?*anyopaque, cond: *std.Io.Condition, wake: std.Io.Condition.Wake) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        self.inner.vtable.conditionWake(self.inner.userdata, cond, wake);
    }

    pub fn now(userdata: ?*anyopaque, clock: std.Io.Clock) std.Io.Clock.Error!std.Io.Timestamp {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.now(self.inner.userdata, clock);
    }

    pub fn sleep(userdata: ?*anyopaque, timeout: std.Io.Timeout) std.Io.SleepError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.sleep(self.inner.userdata, timeout);
    }

    pub fn netListenIp(userdata: ?*anyopaque, address: std.Io.net.IpAddress, options: std.Io.net.IpAddress.ListenOptions) std.Io.net.IpAddress.ListenError!std.Io.net.Server {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.netListenIp(self.inner.userdata, address, options);
    }

    pub fn netAccept(userdata: ?*anyopaque, server: std.Io.net.Socket.Handle) std.Io.net.Server.AcceptError!std.Io.net.Stream {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.netAccept(self.inner.userdata, server);
    }

    pub fn netBindIp(userdata: ?*anyopaque, address: *const std.Io.net.IpAddress, options: std.Io.net.IpAddress.BindOptions) std.Io.net.IpAddress.BindError!std.Io.net.Socket {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.netBindIp(self.inner.userdata, address, options);
    }

    pub fn netConnectIp(userdata: ?*anyopaque, address: *const std.Io.net.IpAddress, options: std.Io.net.IpAddress.ConnectOptions) std.Io.net.IpAddress.ConnectError!std.Io.net.Stream {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.netConnectIp(self.inner.userdata, address, options);
    }

    pub fn netListenUnix(userdata: ?*anyopaque, address: *const std.Io.net.UnixAddress, options: std.Io.net.UnixAddress.ListenOptions) std.Io.net.UnixAddress.ListenError!std.Io.net.Socket.Handle {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.netListenUnix(self.inner.userdata, address, options);
    }

    pub fn netConnectUnix(userdata: ?*anyopaque, address: *const std.Io.net.UnixAddress) std.Io.net.UnixAddress.ConnectError!std.Io.net.Socket.Handle {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.netConnectUnix(self.inner.userdata, address);
    }

    pub fn netSend(userdata: ?*anyopaque, handle: std.Io.net.Socket.Handle, msgs: []std.Io.net.OutgoingMessage, flags: std.Io.net.SendFlags) struct { ?std.Io.net.Socket.SendError, usize } {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.netSend(self.inner.userdata, handle, msgs, flags);
    }

    pub fn netReceive(userdata: ?*anyopaque, handle: std.Io.net.Socket.Handle, message_buffer: []std.Io.net.IncomingMessage, data_buffer: []u8, flags: std.Io.net.ReceiveFlags, timeout: std.Io.Timeout) struct { ?std.Io.net.Socket.ReceiveTimeoutError, usize } {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.netReceive(self.inner.userdata, handle, message_buffer, data_buffer, flags, timeout);
    }

    pub fn netRead(userdata: ?*anyopaque, src: std.Io.net.Socket.Handle, data: [][]u8) std.Io.net.Stream.Reader.Error!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.netRead(self.inner.userdata, src, data);
    }

    pub fn netWrite(userdata: ?*anyopaque, dest: std.Io.net.Socket.Handle, header: []const u8, data: []const []const u8, splat: usize) std.Io.net.Stream.Writer.Error!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.netWrite(self.inner.userdata, dest, header, data, splat);
    }

    pub fn netClose(userdata: ?*anyopaque, handle: std.Io.net.Socket.Handle) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        self.inner.vtable.netClose(self.inner.userdata, handle);
    }

    pub fn netInterfaceNameResolve(userdata: ?*anyopaque, name: *const std.Io.net.Interface.Name) std.Io.net.Interface.Name.ResolveError!std.Io.net.Interface {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.netInterfaceNameResolve(self.inner.userdata, name);
    }

    pub fn netInterfaceName(userdata: ?*anyopaque, iface: std.Io.net.Interface) std.Io.net.Interface.NameError!std.Io.net.Interface.Name {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.netInterfaceName(self.inner.userdata, iface);
    }

    pub fn netLookup(userdata: ?*anyopaque, host: std.Io.net.HostName, q: *std.Io.Queue(std.Io.net.HostName.LookupResult), options: std.Io.net.HostName.LookupOptions) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.netLookup(self.inner.userdata, host, q, options);
    }
};
