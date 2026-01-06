const std = @import("std");

const VFSBase = @import("base.zig").VFSBase;

const log = std.log.scoped(.@"zml/io/vfs/hf");

pub const API = struct {
    const TREESIZE_URL_TEMPLATE = "https://huggingface.co/api/models/{[repo]s}/{[model]s}/treesize/{[rev]s}/{[path]s}";
    const LFS_FILE_URL_TEMPLATE = "https://huggingface.co/{[repo]s}/{[model]s}/resolve/{[rev]s}/{[path]s}";

    pub const TreeSize = struct {
        path: []const u8,
        size: u64,
    };
};

pub const HF = struct {
    pub const Auth = union(enum) {
        none,
        hf_token: []const u8,

        pub const AuthError = error{
            MissingHFToken,
            InvalidHFToken,
            MissingHomePath,
            MissingConfigFile,
            Unexpected,
        };

        pub fn deinit(self: *Auth, allocator: std.mem.Allocator) void {
            switch (self.*) {
                .none => {},
                .hf_token => |token| {
                    allocator.free(token);
                },
            }
        }

        pub fn auto(allocator: std.mem.Allocator, io_: std.Io) AuthError!Auth {
            return fromHomeConfig(allocator, io_) catch |err| switch (err) {
                AuthError.MissingHomePath, AuthError.MissingConfigFile => return fromEnv(allocator) catch |e| switch (e) {
                    AuthError.MissingHFToken => {
                        log.warn("No Hugging Face authentication token found in environment or home config; proceeding without authentication.", .{});
                        return .none;
                    },
                    else => return AuthError.Unexpected,
                },
                else => return AuthError.Unexpected,
            };
        }

        pub fn fromEnv(allocator: std.mem.Allocator) AuthError!Auth {
            const token = std.process.getEnvVarOwned(allocator, "HF_TOKEN") catch |err| switch (err) {
                error.EnvironmentVariableNotFound => return AuthError.MissingHFToken,
                else => return AuthError.Unexpected,
            };
            defer allocator.free(token);

            return .{ .hf_token = std.fmt.allocPrint(allocator, "Bearer {s}", .{token}) catch return AuthError.Unexpected };
        }

        pub fn fromHomeConfig(allocator: std.mem.Allocator, base_io: std.Io) AuthError!Auth {
            const home_path = std.process.getEnvVarOwned(allocator, "HOME") catch |err| switch (err) {
                error.EnvironmentVariableNotFound => return AuthError.MissingHomePath,
                else => return AuthError.Unexpected,
            };
            defer allocator.free(home_path);

            const token_path = std.fmt.allocPrint(allocator, "{s}/.cache/huggingface/token", .{home_path}) catch return AuthError.Unexpected;
            defer allocator.free(token_path);

            const file = std.Io.Dir.openFileAbsolute(base_io, token_path, .{ .mode = .read_only }) catch |err| switch (err) {
                error.FileNotFound => return AuthError.MissingConfigFile,
                else => return AuthError.Unexpected,
            };

            var reader = file.reader(base_io, &.{});
            const size = reader.getSize() catch return AuthError.Unexpected;

            const token = reader.interface.readAlloc(allocator, size) catch return AuthError.Unexpected;
            defer allocator.free(token);

            return .{ .hf_token = std.fmt.allocPrint(allocator, "Bearer {s}", .{std.mem.trim(u8, token, "\n")}) catch return AuthError.Unexpected };
        }
    };

    pub const Repo = struct {
        repo: []const u8,
        model: []const u8,
        rev: []const u8,
        path: []const u8,

        pub fn parse(uri: []const u8) !Repo {
            var parts = std.mem.splitScalar(u8, uri, '/');
            var repo: Repo = .{
                .repo = parts.first(),
                .model = parts.next() orelse return error.InvalidURI,
                .rev = "main",
                .path = std.mem.trimEnd(u8, parts.rest(), "/"),
            };
            if (std.mem.findScalar(u8, repo.model, '@')) |at_index| {
                repo.rev = repo.model[at_index + 1 ..];
                repo.model = repo.model[0..at_index];
            }
            return repo;
        }
    };

    const Handle = struct {
        pub const Type = enum {
            file,
            directory,
        };

        type: Type,
        uri: []const u8,
        pos: u64,
        size: u64,

        pub fn init(allocator: std.mem.Allocator, type_: Type, path: []const u8, size: u64) Handle {
            return .{
                .type = type_,
                .uri = allocator.dupe(u8, path) catch unreachable,
                .pos = 0,
                .size = size,
            };
        }

        pub fn deinit(self: *Handle, allocator: std.mem.Allocator) void {
            allocator.free(self.uri);
        }
    };

    const Config = struct {
        http_client: *std.http.Client,
        auth: Auth = .none,
    };

    allocator: std.mem.Allocator,
    client: *std.http.Client,
    authorization: std.http.Client.Request.Headers.Value = .default,
    handles: std.ArrayList(Handle) = .{},
    closed_handles: std.ArrayList(u32) = .{},
    base: VFSBase,

    pub fn init(allocator: std.mem.Allocator, base_io: std.Io, config: Config) !HF {
        var hf: HF = .{
            .allocator = allocator,
            .base = .init(base_io),
            .client = config.http_client,
        };

        switch (config.auth) {
            .hf_token => |token| {
                hf.authorization = .{ .override = token };
            },
            .none => {},
        }

        return hf;
    }

    pub fn deinit(self: *HF) void {
        self.handles.deinit(self.allocator);
        self.closed_handles.deinit(self.allocator);
    }

    pub fn io(self: *HF) std.Io {
        return .{
            .userdata = &self.base,
            .vtable = &comptime VFSBase.vtable(.{
                .dirOpenDir = dirOpenDir,
                .dirStat = dirStat,
                .dirStatFile = dirStatFile,
                .dirAccess = dirAccess,
                .dirOpenFile = dirOpenFile,
                .dirClose = dirClose,
                .dirRealPath = dirRealPath,
                .dirRealPathFile = dirRealPathFile,
                .fileStat = fileStat,
                .fileLength = fileLength,
                .fileClose = fileClose,
                .fileReadStreaming = fileReadStreaming,
                .fileReadPositional = fileReadPositional,
                .fileSeekBy = fileSeekBy,
                .fileSeekTo = fileSeekTo,
                .fileRealPath = fileRealPath,
            }),
        };
    }

    fn openHandle(self: *HF) !struct { u32, *Handle } {
        if (self.closed_handles.pop()) |idx| {
            return .{ idx, &self.handles.items[idx] };
        }
        return .{ @intCast(self.handles.items.len), try self.handles.addOne(self.allocator) };
    }

    fn closeHandle(self: *HF, idx: u32) !void {
        self.handles.items[idx].deinit(self.allocator);
        try self.closed_handles.append(self.allocator, idx);
    }

    fn getFileHandle(self: *HF, file: std.Io.File) *Handle {
        return &self.handles.items[@intCast(file.handle)];
    }

    fn getDirHandle(self: *HF, dir: std.Io.Dir) *Handle {
        return &self.handles.items[@intCast(dir.handle)];
    }

    fn resolvePath(self: *HF, dir: std.Io.Dir, sub_path: []const u8, out_buffer: []u8) ![]u8 {
        if (std.meta.eql(dir, std.Io.Dir.cwd())) {
            return try std.fmt.bufPrint(out_buffer, "{s}", .{sub_path});
        }

        const handle = self.getDirHandle(dir);
        return try std.fmt.bufPrint(out_buffer, "{s}/{s}", .{ handle.uri, sub_path });
    }

    fn dirOpenDir(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, _: std.Io.Dir.OpenOptions) std.Io.Dir.OpenError!std.Io.Dir {
        const self: *HF = @fieldParentPtr("base", VFSBase.as(userdata));

        const size = self.fetchSize(dir, sub_path) catch |err| {
            log.err("Failed to fetch size for directory with path {s}: {any}", .{ sub_path, err });
            return std.Io.Dir.OpenError.Unexpected;
        };

        var path_buffer: [8 * 1024]u8 = undefined;
        const path = self.resolvePath(dir, sub_path, &path_buffer) catch return std.Io.Dir.OpenError.SystemResources;

        const idx, const handle = self.openHandle() catch return std.Io.Dir.OpenError.Unexpected;
        handle.* = .init(self.allocator, .directory, path, size);

        return .{ .handle = @intCast(idx) };
    }

    fn dirStat(userdata: ?*anyopaque, dir: std.Io.Dir) std.Io.Dir.StatError!std.Io.Dir.Stat {
        const self: *HF = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getDirHandle(dir);

        return .{
            .inode = @intCast(dir.handle),
            .nlink = 0,
            .size = handle.size,
            .permissions = .fromMode(0o444),
            .kind = .directory,
            .atime = null,
            .mtime = std.Io.Timestamp.zero,
            .ctime = std.Io.Timestamp.zero,
        };
    }

    fn dirStatFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, _: std.Io.Dir.StatFileOptions) std.Io.Dir.StatFileError!std.Io.File.Stat {
        const self: *HF = @fieldParentPtr("base", VFSBase.as(userdata));
        const size = self.fetchSize(dir, sub_path) catch return std.Io.Dir.StatFileError.Unexpected;

        return .{
            .inode = @intCast(dir.handle),
            .nlink = 0,
            .size = size,
            .permissions = .fromMode(0o444),
            .kind = .file,
            .atime = null,
            .mtime = std.Io.Timestamp.zero,
            .ctime = std.Io.Timestamp.zero,
        };
    }

    fn dirAccess(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, _: std.Io.Dir.AccessOptions) std.Io.Dir.AccessError!void {
        const self: *HF = @fieldParentPtr("base", VFSBase.as(userdata));
        _ = self.fetchSize(dir, sub_path) catch return std.Io.Dir.AccessError.FileNotFound;
    }

    fn dirOpenFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, _: std.Io.File.OpenFlags) std.Io.File.OpenError!std.Io.File {
        const self: *HF = @fieldParentPtr("base", VFSBase.as(userdata));

        const size = self.fetchSize(dir, sub_path) catch return std.Io.File.OpenError.Unexpected;

        var path_buffer: [8 * 1024]u8 = undefined;
        const path = self.resolvePath(dir, sub_path, &path_buffer) catch return std.Io.File.OpenError.SystemResources;

        const idx, const handle = self.openHandle() catch return std.Io.File.OpenError.Unexpected;
        handle.* = .init(self.allocator, .file, path, size);

        return .{ .handle = @intCast(idx) };
    }

    fn dirClose(userdata: ?*anyopaque, dirs: []const std.Io.Dir) void {
        const self: *HF = @fieldParentPtr("base", VFSBase.as(userdata));
        for (dirs) |dir| {
            self.closeHandle(@intCast(dir.handle)) catch unreachable;
        }
    }

    fn dirRealPath(userdata: ?*anyopaque, dir: std.Io.Dir, out_buffer: []u8) std.Io.Dir.RealPathError!usize {
        const self: *HF = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getDirHandle(dir);
        const len = handle.uri.len;
        @memcpy(out_buffer[0..len], handle.uri);
        return len;
    }

    fn dirRealPathFile(userdata: ?*anyopaque, dir: std.Io.Dir, path_name: []const u8, out_buffer: []u8) std.Io.Dir.RealPathFileError!usize {
        const self: *HF = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getDirHandle(dir);
        const full_path = std.fmt.bufPrint(out_buffer, "{s}/{s}", .{ handle.uri, path_name }) catch return std.Io.Dir.RealPathFileError.NameTooLong;
        return full_path.len;
    }

    fn fileStat(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.StatError!std.Io.File.Stat {
        const self: *HF = @fieldParentPtr("base", VFSBase.as(userdata));

        const handle = self.getFileHandle(file);

        return .{
            .inode = @intCast(file.handle),
            .nlink = 0,
            .size = handle.size,
            .permissions = .fromMode(0o444),
            .kind = .file,
            .atime = null,
            .mtime = std.Io.Timestamp.zero,
            .ctime = std.Io.Timestamp.zero,
        };
    }

    fn fileLength(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.LengthError!u64 {
        const self: *HF = @fieldParentPtr("base", VFSBase.as(userdata));
        return self.getFileHandle(file).size;
    }

    fn fileClose(userdata: ?*anyopaque, files: []const std.Io.File) void {
        const self: *HF = @fieldParentPtr("base", VFSBase.as(userdata));
        for (files) |file| {
            self.closeHandle(@intCast(file.handle)) catch unreachable;
        }
    }

    fn fileReadStreaming(userdata: ?*anyopaque, file: std.Io.File, data: []const []u8) std.Io.File.Reader.Error!usize {
        const self: *HF = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        const total = self.performRead(handle, data, handle.pos) catch |err| {
            log.err("Failed to perform read for file {s} at pos {d}: {any}", .{ handle.uri, handle.pos, err });
            return std.Io.File.Reader.Error.Unexpected;
        };
        handle.pos += @intCast(total);
        return total;
    }

    fn fileReadPositional(userdata: ?*anyopaque, file: std.Io.File, data: []const []u8, offset: u64) std.Io.File.ReadPositionalError!usize {
        const self: *HF = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        return self.performRead(handle, data, offset) catch |err| {
            log.err("Failed to perform read for file {s} at pos {d}: {any}", .{ handle.uri, offset, err });
            return std.Io.File.Reader.Error.Unexpected;
        };
    }

    fn fileSeekBy(userdata: ?*anyopaque, file: std.Io.File, relative_offset: i64) std.Io.File.SeekError!void {
        const self: *HF = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);

        handle.pos = if (relative_offset >= 0)
            handle.pos + @as(u64, @intCast(relative_offset))
        else
            handle.pos - @as(u64, @intCast(-relative_offset));
    }

    fn fileSeekTo(userdata: ?*anyopaque, file: std.Io.File, absolute_offset: u64) std.Io.File.SeekError!void {
        const self: *HF = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        handle.pos = absolute_offset;
    }

    fn fileRealPath(userdata: ?*anyopaque, file: std.Io.File, out_buffer: []u8) std.Io.File.RealPathError!usize {
        const self: *HF = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        const len = handle.uri.len;
        @memcpy(out_buffer[0..len], handle.uri);
        return len;
    }

    fn fetchSize(self: *HF, dir: std.Io.Dir, sub_path: []const u8) !u64 {
        var path_buffer: [8 * 1024]u8 = undefined;
        var url_buffer: [8 * 1024]u8 = undefined;
        var redirect_buffer: [8 * 1024]u8 = undefined;

        const path = try self.resolvePath(dir, sub_path, &path_buffer);

        const repo = Repo.parse(path) catch return std.Io.File.OpenError.BadPathName;
        const url: []const u8 = try std.fmt.bufPrint(&url_buffer, API.TREESIZE_URL_TEMPLATE, repo);
        const uri = std.Uri.parse(url) catch return std.Io.Dir.OpenError.BadPathName;

        var req = try self.client.request(.GET, uri, .{
            .headers = .{ .authorization = self.authorization },
        });
        defer req.deinit();

        try req.sendBodiless();

        var res = try req.receiveHead(&redirect_buffer);

        if (res.head.status != .ok) {
            log.err("Failed to fetch tree size for {s}", .{url});
            log.err("{s}", .{res.head.bytes});
        }

        const body = try res.reader(&.{}).readAlloc(self.allocator, res.head.content_length.?);
        defer self.allocator.free(body);

        if (res.head.status != .ok) {
            log.err("{s}", .{body});
            return error.RequestFailed;
        }

        const parsed = try std.json.parseFromSlice(
            API.TreeSize,
            self.allocator,
            body,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        return parsed.value.size;
    }

    fn performRead(self: *HF, handle: *Handle, data: []const []u8, offset: u64) !usize {
        if (offset >= handle.size) return 0;

        var range_buf: [64]u8 = undefined;
        const range_header = blk: {
            var total_bytes: u64 = 0;
            for (data) |buf| {
                total_bytes += @as(u64, buf.len);
            }
            const remaining = handle.size - offset;
            const take = @min(remaining, total_bytes);
            const end = offset + take - 1;
            break :blk std.fmt.bufPrint(&range_buf, "bytes={d}-{d}", .{ offset, end }) catch unreachable;
        };

        const repo: Repo = try .parse(handle.uri);
        var url_buffer: [8 * 1024]u8 = undefined;
        const url: []const u8 = try std.fmt.bufPrint(&url_buffer, API.LFS_FILE_URL_TEMPLATE, repo);

        const uri: std.Uri = try .parse(url);
        var req = try self.client.request(.GET, uri, .{
            .headers = .{ .authorization = self.authorization },
            .extra_headers = &.{.{ .name = "Range", .value = range_header }},
        });
        defer req.deinit();

        try req.sendBodiless();

        var redirect_buffer: [8 * 1024]u8 = undefined;
        var res = try req.receiveHead(&redirect_buffer);

        if (res.head.status != .partial_content) {
            log.err("Failed to perform read for {s}", .{url});
            log.err("{s}", .{res.head.bytes});
            return error.RequestFailed;
        }

        const reader = res.reader(&.{});

        const content_range = blk: {
            var it = res.head.iterateHeaders();
            while (it.next()) |header| {
                if (std.ascii.eqlIgnoreCase(header.name, "Content-Range")) {
                    break :blk parseContentRange(header.value);
                }
            }
            break :blk null;
        };

        if (content_range) |range| {
            if (range.start < offset) {
                try reader.discardAll(offset - range.start);
            }
        }

        return try reader.readSliceShort(data[0]);
    }

    const ContentRange = struct {
        start: u64,
        end: u64,
        total: u64,
    };

    fn parseContentRange(value: []const u8) ?ContentRange {
        const space = std.mem.indexOfScalar(u8, value, ' ') orelse return null;
        const dash = std.mem.indexOfScalar(u8, value, '-') orelse return null;
        const slash = std.mem.indexOfScalar(u8, value, '/') orelse return null;

        return .{
            .start = std.fmt.parseInt(u64, value[space + 1 .. dash], 10) catch return null,
            .end = std.fmt.parseInt(u64, value[dash + 1 .. slash], 10) catch return null,
            .total = std.fmt.parseInt(u64, value[slash + 1 ..], 10) catch return null,
        };
    }
};
