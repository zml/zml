const std = @import("std");

const stdx = @import("stdx");

const VFSBase = @import("base.zig").VFSBase;

const log = std.log.scoped(.@"zml/io/vfs/hf");

pub const API = struct {
    const TREE_URL_TEMPLATE = "https://huggingface.co/api/models/{[repo]s}/{[model]s}/tree/{[rev]s}/{[path]s}?expand=false&recursive=true&limit=1000";
    const LFS_FILE_URL_TEMPLATE = "https://huggingface.co/{[repo]s}/{[model]s}/resolve/{[rev]s}/{[path]s}";

    pub const Tree = struct {
        oid: []const u8,
        path: []const u8,
        type: []const u8,
        size: u64,
    };
};

const ReadState = struct { children: []const TreeNode, index: usize };

pub const TreeNode = struct {
    name: []const u8,
    kind: std.Io.File.Kind,
    size: u64,
    children: ?std.ArrayList(TreeNode),

    fn deinit(self: *TreeNode, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        if (self.children) |*children| {
            for (children.items) |*child| child.deinit(allocator);
            children.deinit(allocator);
        }
    }
};

const RepoKey = struct {
    repo: []const u8,
    model: []const u8,
    rev: []const u8,

    fn format(self: RepoKey, buffer: []u8) ![]u8 {
        return std.fmt.bufPrint(buffer, "{s}/{s}@{s}", .{ self.repo, self.model, self.rev });
    }

    fn allocFormat(self: RepoKey, allocator: std.mem.Allocator) ![]u8 {
        return std.fmt.allocPrint(allocator, "{s}/{s}@{s}", .{ self.repo, self.model, self.rev });
    }
};

pub const HF = struct {
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

        fn toKey(self: Repo) RepoKey {
            return .{ .repo = self.repo, .model = self.model, .rev = self.rev };
        }
    };

    const Handle = struct {
        pub const Type = enum { file, directory };

        type: Type,
        uri: []const u8,
        pos: u64,
        size: u64,

        pub fn init(allocator: std.mem.Allocator, handle_type: Type, uri: []const u8, size: u64) !Handle {
            return .{
                .type = handle_type,
                .uri = try allocator.dupe(u8, uri),
                .pos = 0,
                .size = size,
            };
        }

        pub fn deinit(self: *Handle, allocator: std.mem.Allocator) void {
            allocator.free(self.uri);
        }
    };

    allocator: std.mem.Allocator,
    mutex: std.Io.Mutex = .init,
    client: *std.http.Client,
    authorization: std.http.Client.Request.Headers.Value,
    handles: stdx.SegmentedList(Handle, 0) = .{},
    closed_handles: std.ArrayList(u32) = .{},
    base: VFSBase,
    trees: std.StringHashMapUnmanaged(std.ArrayList(TreeNode)) = .{},
    dir_read_states: std.AutoHashMapUnmanaged(*std.Io.Dir.Reader, ReadState) = .{},

    pub fn init(allocator: std.mem.Allocator, inner: std.Io, http_client: *std.http.Client, hf_token: ?[]const u8) !HF {
        return .{
            .allocator = allocator,
            .base = .init(inner),
            .client = http_client,
            .authorization = if (hf_token) |token| blk: {
                break :blk .{
                    .override = try std.fmt.allocPrint(allocator, "Bearer {s}", .{std.mem.trim(u8, token, " \t\n\r")}),
                };
            } else blk: {
                break :blk .default;
            },
        };
    }

    pub fn auto(allocator: std.mem.Allocator, inner: std.Io, http_client: *std.http.Client, environ_map: *std.process.Environ.Map) !HF {
        const hf_token = if (environ_map.get("HF_TOKEN")) |token| blk: {
            break :blk token;
        } else blk: {
            const home_path = environ_map.get("HOME") orelse break :blk null;

            var path_buf: [256]u8 = undefined;
            const token_path = std.fmt.bufPrint(&path_buf, "{s}/.cache/huggingface/token", .{home_path}) catch break :blk null;

            var file = std.Io.Dir.openFileAbsolute(inner, token_path, .{ .mode = .read_only }) catch break :blk null;
            defer file.close(inner);

            const size = file.stat(inner) catch break :blk null;
            var reader = file.reader(inner, &.{});
            const token = reader.interface.readAlloc(allocator, size.size) catch break :blk null;

            break :blk token;
        };

        if (hf_token == null) log.warn("No Hugging Face authentication token found in environment or home config; proceeding without authentication.", .{});

        return init(allocator, inner, http_client, hf_token);
    }

    pub fn deinit(self: *HF) void {
        var idx: usize = 0;
        while (idx < self.handles.len) : (idx += 1) {
            const is_closed = for (self.closed_handles.items) |closed_idx| {
                if (closed_idx == idx) break true;
            } else false;

            if (!is_closed) {
                self.handles.at(idx).deinit(self.allocator);
            }
        }
        self.handles.deinit(self.allocator);
        self.closed_handles.deinit(self.allocator);

        var it = self.trees.iterator();
        while (it.next()) |entry| {
            for (entry.value_ptr.items) |*node| {
                node.deinit(self.allocator);
            }
            entry.value_ptr.deinit(self.allocator);
            self.allocator.free(entry.key_ptr.*);
        }
        self.trees.deinit(self.allocator);
        self.dir_read_states.deinit(self.allocator);

        switch (self.authorization) {
            .default, .omit => {},
            .override => |t| self.allocator.free(t),
        }
    }

    pub fn io(self: *HF) std.Io {
        return .{
            .userdata = &self.base,
            .vtable = &comptime VFSBase.vtable(.{
                .operate = operate,
                .dirOpenDir = dirOpenDir,
                .dirStat = dirStat,
                .dirStatFile = dirStatFile,
                .dirAccess = dirAccess,
                .dirOpenFile = dirOpenFile,
                .dirClose = dirClose,
                .dirRead = dirRead,
                .dirRealPath = dirRealPath,
                .dirRealPathFile = dirRealPathFile,
                .fileStat = fileStat,
                .fileLength = fileLength,
                .fileClose = fileClose,
                .fileReadPositional = fileReadPositional,
                .fileSeekBy = fileSeekBy,
                .fileSeekTo = fileSeekTo,
                .fileRealPath = fileRealPath,
            }),
        };
    }

    fn openHandle(self: *HF) !struct { u32, *Handle } {
        self.mutex.lockUncancelable(self.base.inner);
        defer self.mutex.unlock(self.base.inner);

        if (self.closed_handles.pop()) |idx| {
            return .{ idx, self.handles.at(idx) };
        }
        const len: u32 = @intCast(self.handles.len);
        return .{ len, try self.handles.addOne(self.allocator) };
    }

    fn closeHandle(self: *HF, idx: u32) !void {
        self.mutex.lockUncancelable(self.base.inner);
        defer self.mutex.unlock(self.base.inner);

        self.handles.at(idx).deinit(self.allocator);
        try self.closed_handles.append(self.allocator, idx);
    }

    fn getFileHandle(self: *HF, file: std.Io.File) *Handle {
        self.mutex.lockUncancelable(self.base.inner);
        defer self.mutex.unlock(self.base.inner);
        return self.handles.at(@intCast(file.handle));
    }

    fn getDirHandle(self: *HF, dir: std.Io.Dir) *Handle {
        self.mutex.lockUncancelable(self.base.inner);
        defer self.mutex.unlock(self.base.inner);
        return self.handles.at(@intCast(dir.handle));
    }

    fn resolvePath(self: *HF, dir: std.Io.Dir, sub_path: []const u8, out_buffer: []u8) ![]u8 {
        if (std.meta.eql(dir, std.Io.Dir.cwd())) {
            return try std.fmt.bufPrint(out_buffer, "{s}", .{sub_path});
        }

        const handle = self.getDirHandle(dir);

        const trimmed_uri = std.mem.trimEnd(u8, handle.uri, "/");
        const trimmed_sub_path = std.mem.trimStart(u8, sub_path, "/");

        if (trimmed_uri.len == 0) return try std.fmt.bufPrint(out_buffer, "{s}", .{trimmed_sub_path});
        if (trimmed_sub_path.len == 0) return try std.fmt.bufPrint(out_buffer, "{s}", .{trimmed_uri});
        return try std.fmt.bufPrint(out_buffer, "{s}/{s}", .{ trimmed_uri, trimmed_sub_path });
    }

    fn getOrFetchTree(self: *HF, repo: Repo) !*std.ArrayList(TreeNode) {
        const key = repo.toKey();

        var cache_key_buffer: [512]u8 = undefined;

        const cache_key_lookup = try key.format(&cache_key_buffer);

        if (self.trees.getPtr(cache_key_lookup)) |tree_ptr| return tree_ptr;

        const cache_key = try key.allocFormat(self.allocator);
        errdefer self.allocator.free(cache_key);

        self.mutex.lockUncancelable(self.base.inner);
        defer self.mutex.unlock(self.base.inner);

        const gop = try self.trees.getOrPut(self.allocator, cache_key);
        errdefer _ = self.trees.remove(cache_key);

        if (gop.found_existing) {
            self.allocator.free(cache_key);
            return gop.value_ptr;
        }

        const tree_root = try self.fetchTreeFromAPI(repo);
        gop.value_ptr.* = tree_root;

        return gop.value_ptr;
    }

    fn fetchTreeFromAPI(self: *HF, repo: Repo) !std.ArrayList(TreeNode) {
        var url_buffer: [8 * 1024]u8 = undefined;
        var redirect_buffer: [8 * 1024]u8 = undefined;

        const url = try std.fmt.bufPrint(&url_buffer, API.TREE_URL_TEMPLATE, .{
            .repo = repo.repo,
            .model = repo.model,
            .rev = repo.rev,
            .path = "",
        });
        const uri = try std.Uri.parse(url);

        var req = try self.client.request(.GET, uri, .{
            .headers = .{ .authorization = self.authorization },
        });
        defer req.deinit();

        try req.sendBodiless();

        var res = try req.receiveHead(&redirect_buffer);

        if (res.head.status != .ok) {
            log.err("Failed to fetch tree: status={}", .{res.head.status});
            return error.RequestFailed;
        }

        const body = try res.reader(&.{}).readAlloc(self.allocator, res.head.content_length.?);
        defer self.allocator.free(body);

        const parsed = try std.json.parseFromSlice(
            []API.Tree,
            self.allocator,
            body,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        var tree_root: std.ArrayList(TreeNode) = .{};
        for (parsed.value) |item| {
            try insertTreeNode(self.allocator, &tree_root, item);
        }

        return tree_root;
    }

    fn findDirChildren(tree_items: []const TreeNode, dir_path: []const u8) ?[]const TreeNode {
        if (dir_path.len == 0) {
            return tree_items;
        }

        var parts = std.mem.tokenizeScalar(u8, dir_path, '/');
        var current = tree_items;

        while (parts.next()) |part| {
            var found: ?*const TreeNode = null;
            for (current) |*node| {
                if (std.mem.eql(u8, node.name, part)) {
                    found = node;
                    break;
                }
            }

            if (found) |node| {
                if (node.children) |children| {
                    current = children.items;
                } else {
                    return null;
                }
            } else {
                return null;
            }
        }

        return current;
    }

    fn findNode(tree_items: []const TreeNode, path: []const u8) ?*const TreeNode {
        if (path.len == 0) return null;

        var parts = std.mem.tokenizeScalar(u8, path, '/');
        var current = tree_items;

        while (parts.next()) |part| {
            const is_last = parts.peek() == null;

            var found: ?*const TreeNode = null;
            for (current) |*node| {
                if (std.mem.eql(u8, node.name, part)) {
                    found = node;
                    break;
                }
            }

            if (found) |node| {
                if (is_last) return node;
                if (node.children) |children| {
                    current = children.items;
                } else {
                    return null;
                }
            } else {
                return null;
            }
        }

        return null;
    }

    fn getSizeFromTree(tree: *std.ArrayList(TreeNode), path: []const u8) ?u64 {
        if (path.len == 0) {
            var total: u64 = 0;
            for (tree.items) |*node| {
                if (node.kind == .file) total += node.size;
            }
            return total;
        }

        const node = findNode(tree.items, path) orelse return null;
        return node.size;
    }

    fn insertTreeNode(allocator: std.mem.Allocator, root: *std.ArrayList(TreeNode), item: API.Tree) !void {
        var parts = std.mem.tokenizeScalar(u8, item.path, '/');
        var current_list = root;

        var parents: std.ArrayList(*TreeNode) = .{};
        defer parents.deinit(allocator);

        while (true) {
            const part = parts.next() orelse break;
            const is_last = parts.peek() == null;

            var found: ?*TreeNode = null;
            for (current_list.items) |*node| {
                if (std.mem.eql(u8, node.name, part)) {
                    found = node;
                    break;
                }
            }

            if (found) |f| {
                if (!is_last) {
                    try parents.append(allocator, f);
                    if (f.children == null) {
                        f.children = .{};
                    }
                    current_list = &f.children.?;
                } else if (std.mem.eql(u8, item.type, "file")) {
                    f.size = item.size;
                }
            } else {
                const is_file = is_last and std.mem.eql(u8, item.type, "file");
                const new_node: TreeNode = .{
                    .name = try allocator.dupe(u8, part),
                    .kind = if (is_file) .file else .directory,
                    .size = if (is_last) item.size else 0,
                    .children = if (is_file) null else .{},
                };
                try current_list.append(allocator, new_node);

                if (!is_last) {
                    const new_node_ptr = &current_list.items[current_list.items.len - 1];
                    try parents.append(allocator, new_node_ptr);
                    current_list = &new_node_ptr.children.?;
                }
            }
        }

        for (parents.items) |parent| {
            parent.size += item.size;
        }
    }

    fn operate(userdata: ?*anyopaque, operation: std.Io.Operation) std.Io.Cancelable!std.Io.Operation.Result {
        const self: *HF = @fieldParentPtr("base", VFSBase.as(userdata));
        switch (operation) {
            .file_read_streaming => |o| {
                const handle = self.getFileHandle(o.file);
                const total = self.performRead(handle, o.data, handle.pos) catch |err| {
                    log.err("Failed to perform read for file {s} at pos {d}: {any}", .{ handle.uri, handle.pos, err });
                    return .{ .file_read_streaming = std.Io.File.ReadStreamingError.EndOfStream };
                };

                if (total == 0) {
                    return .{ .file_read_streaming = std.Io.File.ReadStreamingError.EndOfStream };
                }

                handle.pos += @intCast(total);
                return .{ .file_read_streaming = total };
            },
            .file_write_streaming, .device_io_control => {
                return self.base.inner.vtable.operate(self.base.inner.userdata, operation);
            },
        }
    }

    fn dirOpenDir(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, _: std.Io.Dir.OpenOptions) std.Io.Dir.OpenError!std.Io.Dir {
        const self: *HF = @fieldParentPtr("base", VFSBase.as(userdata));

        var path_buffer: [8 * 1024]u8 = undefined;
        const full_path = self.resolvePath(dir, sub_path, &path_buffer) catch return std.Io.Dir.OpenError.SystemResources;

        const repo = Repo.parse(full_path) catch return std.Io.Dir.OpenError.BadPathName;
        const tree = self.getOrFetchTree(repo) catch return std.Io.Dir.OpenError.Unexpected;
        const size = getSizeFromTree(tree, repo.path) orelse return std.Io.Dir.OpenError.FileNotFound;

        const idx, const handle = self.openHandle() catch return std.Io.Dir.OpenError.Unexpected;
        handle.* = Handle.init(self.allocator, .directory, full_path, size) catch return std.Io.Dir.OpenError.Unexpected;

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
            .block_size = 1,
        };
    }

    fn dirStatFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, _: std.Io.Dir.StatFileOptions) std.Io.Dir.StatFileError!std.Io.File.Stat {
        const self: *HF = @fieldParentPtr("base", VFSBase.as(userdata));

        var path_buffer: [8 * 1024]u8 = undefined;
        const full_path = self.resolvePath(dir, sub_path, &path_buffer) catch return std.Io.Dir.StatFileError.SystemResources;

        const repo = Repo.parse(full_path) catch return std.Io.Dir.StatFileError.Unexpected;
        const tree = self.getOrFetchTree(repo) catch return std.Io.Dir.StatFileError.Unexpected;
        const node = findNode(tree.items, repo.path) orelse return std.Io.Dir.StatFileError.IsDir;

        if (node.kind == .directory) return std.Io.Dir.StatFileError.IsDir;
        const size = getSizeFromTree(tree, repo.path) orelse return std.Io.Dir.StatFileError.FileNotFound;

        return .{
            .inode = @intCast(0),
            .nlink = 0,
            .size = size,
            .permissions = .fromMode(0o444),
            .kind = .file,
            .atime = null,
            .mtime = std.Io.Timestamp.zero,
            .ctime = std.Io.Timestamp.zero,
            .block_size = 1,
        };
    }

    fn dirAccess(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, _: std.Io.Dir.AccessOptions) std.Io.Dir.AccessError!void {
        const self: *HF = @fieldParentPtr("base", VFSBase.as(userdata));

        var path_buffer: [8 * 1024]u8 = undefined;
        const full_path = self.resolvePath(dir, sub_path, &path_buffer) catch return std.Io.Dir.AccessError.SystemResources;

        const repo = Repo.parse(full_path) catch return std.Io.Dir.AccessError.FileNotFound;
        const tree = self.getOrFetchTree(repo) catch return std.Io.Dir.AccessError.Unexpected;

        if (repo.path.len == 0) return;

        const node = findNode(tree.items, repo.path);
        if (node == null) return std.Io.Dir.AccessError.FileNotFound;
    }

    fn dirOpenFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, _: std.Io.File.OpenFlags) std.Io.File.OpenError!std.Io.File {
        const self: *HF = @fieldParentPtr("base", VFSBase.as(userdata));

        var path_buffer: [8 * 1024]u8 = undefined;
        const full_path = self.resolvePath(dir, sub_path, &path_buffer) catch return std.Io.File.OpenError.SystemResources;

        const repo = Repo.parse(full_path) catch return std.Io.File.OpenError.BadPathName;
        const tree = self.getOrFetchTree(repo) catch return std.Io.File.OpenError.Unexpected;
        const size = getSizeFromTree(tree, repo.path) orelse return std.Io.File.OpenError.FileNotFound;

        const idx, const handle = self.openHandle() catch return std.Io.File.OpenError.Unexpected;
        handle.* = Handle.init(self.allocator, .file, full_path, size) catch return std.Io.File.OpenError.Unexpected;

        return .{ .handle = @intCast(idx), .flags = .{ .nonblocking = false } };
    }

    fn dirClose(userdata: ?*anyopaque, dirs: []const std.Io.Dir) void {
        const self: *HF = @fieldParentPtr("base", VFSBase.as(userdata));
        for (dirs) |d| {
            self.closeHandle(@intCast(d.handle)) catch {};
        }
    }

    fn dirRead(userdata: ?*anyopaque, reader: *std.Io.Dir.Reader, entries: []std.Io.Dir.Entry) std.Io.Dir.Reader.Error!usize {
        const self: *HF = @fieldParentPtr("base", VFSBase.as(userdata));

        if (reader.state == .finished) return 0;

        if (reader.state == .reset) {
            _ = self.dir_read_states.remove(reader);

            const handle = self.getDirHandle(reader.dir);
            const repo = Repo.parse(handle.uri) catch return std.Io.Dir.Reader.Error.Unexpected;
            const tree = self.getOrFetchTree(repo) catch return std.Io.Dir.Reader.Error.Unexpected;

            const children = findDirChildren(tree.items, repo.path) orelse &.{};

            self.dir_read_states.put(self.allocator, reader, .{
                .children = children,
                .index = 0,
            }) catch return std.Io.Dir.Reader.Error.Unexpected;

            reader.state = if (children.len > 0) .reading else .finished;
            if (children.len == 0) return 0;
        }

        const state = self.dir_read_states.getPtr(reader) orelse return std.Io.Dir.Reader.Error.Unexpected;

        var count: usize = 0;
        while (count < entries.len and state.index < state.children.len) {
            const node = &state.children[state.index];
            entries[count] = .{
                .name = node.name,
                .kind = node.kind,
                .inode = state.index,
            };
            count += 1;
            state.index += 1;
        }

        if (state.index >= state.children.len) {
            reader.state = .finished;
            _ = self.dir_read_states.remove(reader);
        }

        return count;
    }

    fn dirRealPath(userdata: ?*anyopaque, dir: std.Io.Dir, out_buffer: []u8) std.Io.Dir.RealPathError!usize {
        const self: *HF = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getDirHandle(dir);
        const path = std.fmt.bufPrint(out_buffer, "{s}", .{handle.uri}) catch return std.Io.Dir.RealPathError.SystemResources;
        return path.len;
    }

    fn dirRealPathFile(userdata: ?*anyopaque, dir: std.Io.Dir, path_name: []const u8, out_buffer: []u8) std.Io.Dir.RealPathFileError!usize {
        const self: *HF = @fieldParentPtr("base", VFSBase.as(userdata));
        const real_path = self.resolvePath(dir, path_name, out_buffer) catch return std.Io.Dir.RealPathFileError.NameTooLong;
        return real_path.len;
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
            .block_size = 1,
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
        const path = std.fmt.bufPrint(out_buffer, "{s}", .{handle.uri}) catch return std.Io.File.RealPathError.SystemResources;
        return path.len;
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

        const content_range = blk: {
            var it = res.head.iterateHeaders();
            while (it.next()) |header| {
                if (std.ascii.eqlIgnoreCase(header.name, "Content-Range")) {
                    break :blk parseContentRange(header.value);
                }
            }
            break :blk null;
        };

        const reader = res.reader(&.{});

        if (content_range) |cr| {
            if (cr.start < offset) {
                try reader.discardAll(offset - cr.start);
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
