const std = @import("std");

const stdx = @import("stdx");

const VFSBase = @import("base.zig").VFSBase;

const log = std.log.scoped(.@"zml/io/vfs/hf");

pub const API = struct {
    const TREE_URL_TEMPLATE = "https://huggingface.co/api/models/{[repo]s}/{[model]s}/tree/{[rev]s}/{[path]s}";
    const TREESIZE_URL_TEMPLATE = "https://huggingface.co/api/models/{[repo]s}/{[model]s}/treesize/{[rev]s}/{[path]s}";
    const RAW_FILE_URL_REMPLATE = "https://huggingface.co/{model}/raw/{rev}/{path}";
    const LFS_FILE_URL_TEMPLATE = "https://huggingface.co/{[repo]s}/{[model]s}/resolve/{[rev]s}/{[path]s}";

    pub const Entry = struct {
        pub const Type = enum {
            file,
            directory,
        };

        pub const LFS = struct {
            oid: []const u8,
            size: u64,
            pointerSize: u64,
        };

        type: Type,
        oid: []const u8,
        size: u64,
        path: []const u8,
        lfs: ?LFS = null,
    };
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
    };

    pub const Handle = struct {
        pub const Type = enum {
            file,
            directory,
        };

        type: Type,
        url: []const u8,
        uri: std.Uri,
        pos: u64,
        size: u64,

        pub fn init(allocator: std.mem.Allocator, type_: Type, sub_path: []const u8, size: u64) Handle {
            const repo = Repo.parse(sub_path) catch unreachable;
            const lfs_url = std.fmt.allocPrint(allocator, API.LFS_FILE_URL_TEMPLATE, repo) catch unreachable;
            errdefer allocator.free(lfs_url);

            return .{
                .type = type_,
                .url = lfs_url,
                .uri = std.Uri.parse(lfs_url) catch unreachable,
                .pos = 0,
                .size = size,
            };
        }

        pub fn deinit(self: *Handle, allocator: std.mem.Allocator) void {
            allocator.free(self.url);
        }
    };

    base: VFSBase,
    authorization: std.http.Client.Request.Headers.Value = .default,
    allocator: std.mem.Allocator,
    client: *std.http.Client,
    handles: std.ArrayList(Handle) = .{},
    closed_handles: std.ArrayList(u32) = .{},

    pub fn init(allocator: std.mem.Allocator, io_: std.Io, http_client_: *std.http.Client) !HF {
        return .{
            .allocator = allocator,
            .base = .init(io_),
            .client = http_client_,
            .authorization = .{
                .override = "Bearer xxxx",
            },
        };
    }

    pub fn deinit(self: *HF) void {
        _ = self; // autofix
    }

    pub fn io(self: *HF) std.Io {
        return .{
            .userdata = &self.base,
            .vtable = &comptime VFSBase.vtable(.{
                .dirOpenFile = dirOpenFile,
                .fileReadPositional = fileReadPositional,
                .fileReadStreaming = fileReadStreaming,
                .fileClose = fileClose,
            }),
        };
    }

    pub fn openHandle(self: *HF) !struct { u32, *Handle } {
        if (self.closed_handles.pop()) |idx| {
            return .{ idx, &self.handles.items[idx] };
        }
        return .{ @intCast(self.handles.items.len), try self.handles.addOne(self.allocator) };
    }

    pub fn closeHandle(self: *HF, idx: u32) !void {
        try self.closed_handles.append(self.allocator, idx);
    }

    fn getFileHandle(self: *HF, file: std.Io.File) *Handle {
        return &self.handles.items[@intCast(file.handle)];
    }

    fn parseContentRange(value: []const u8) struct { u64, u64, u64 } {
        const space = std.mem.indexOfScalar(u8, value, ' ').?;
        const range = std.mem.indexOfScalar(u8, value, '-').?;
        const slash = std.mem.indexOfScalar(u8, value, '/').?;

        return .{
            std.fmt.parseInt(u64, value[space + 1 .. range], 10) catch 0,
            std.fmt.parseInt(u64, value[range + 1 .. slash], 10) catch 0,
            std.fmt.parseInt(u64, value[slash + 1 ..], 10) catch 0,
        };
    }

    fn dirOpenFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, flags: std.Io.File.OpenFlags) std.Io.File.OpenError!std.Io.File {
        _ = dir; // autofix
        _ = flags; // autofix
        // _ = flags; // autofix
        const self: *HF = @fieldParentPtr("base", VFSBase.as(userdata));

        var url_buffer: [1024]u8 = undefined;
        var redirect_buffer: [8 * 1024]u8 = undefined;
        var aux_buffer: []u8 = &redirect_buffer;

        const repo = Repo.parse(sub_path) catch return std.Io.File.OpenError.BadPathName;
        const url: []const u8 = std.fmt.bufPrint(&url_buffer, API.LFS_FILE_URL_TEMPLATE, repo) catch return std.Io.File.OpenError.BadPathName;

        log.info("Opening url: {s}", .{url});

        var uri = std.Uri.parse(url) catch return std.Io.File.OpenError.BadPathName;
        while (true) {
            var req = self.client.request(.HEAD, uri, .{
                .headers = .{ .authorization = self.authorization },
                .redirect_behavior = .not_allowed,
            }) catch return std.Io.File.Reader.Error.Unexpected;
            defer req.deinit();

            req.sendBodiless() catch {
                log.err("Failed to send HTTP request for HF model size API URL", .{});
                return std.Io.File.Reader.Error.Unexpected;
            };

            var resp = req.receiveHead(&.{}) catch {
                log.err("Failed to receive HTTP response for HF model size API URL", .{});
                return std.Io.File.Reader.Error.Unexpected;
            };

            if (resp.head.status.class() == .redirect) {
                const location = resp.head.location.?;
                @memcpy(aux_buffer[0..location.len], location);
                uri = uri.resolveInPlace(location.len, &aux_buffer) catch unreachable;
                log.info("Redirect {f}", .{uri});
                continue;
            }

            const idx, const handle = self.openHandle() catch return std.Io.File.OpenError.Unexpected;
            handle.* = .init(self.allocator, .file, sub_path, resp.head.content_length.?);

            return .{ .handle = @intCast(idx) };
        }
    }

    pub fn performRead(self: *HF, handle: *Handle, data: []const []u8, offset: u64) std.Io.File.Reader.Error!usize {
        if (offset >= handle.size) {
            log.info("EOF", .{});
            return 0;
        }

        var range_buf: [64]u8 = undefined;
        const range_header = blk: {
            var total_bytes: u64 = 0;
            for (data) |buf| {
                total_bytes += buf.len;
            }
            break :blk std.fmt.bufPrint(&range_buf, "bytes={d}-{d}", .{ offset, @min(handle.size - offset, offset + total_bytes) }) catch unreachable;
        };

        var req = self.client.request(.GET, handle.uri, .{
            .headers = .{ .authorization = self.authorization },
            .extra_headers = &.{
                .{ .name = "Range", .value = range_header },
            },
        }) catch return std.Io.File.Reader.Error.Unexpected;
        defer req.deinit();

        req.sendBodiless() catch {
            log.err("Failed to send HTTP request for HF model size API URL", .{});
            return std.Io.File.Reader.Error.Unexpected;
        };

        var redirect_buffer: [8 * 1024]u8 = undefined;
        var resp = req.receiveHead(&redirect_buffer) catch {
            log.err("Failed to receive HTTP response for HF model size API URL", .{});
            return std.Io.File.Reader.Error.Unexpected;
        };

        const range = blk: {
            var it = resp.head.iterateHeaders();
            while (it.next()) |header| {
                if (std.ascii.eqlIgnoreCase(header.name, "Content-Range")) {
                    break :blk parseContentRange(header.value);
                }
            } else break :blk .{ 0, 0, 0 };
        };

        log.info(">>>>> {any}", .{range});

        const reader = resp.reader(&.{});

        if (range.@"0" < offset) {
            reader.discardAll(offset - range.@"0") catch return std.Io.File.Reader.Error.Unexpected;
        }

        return reader.readSliceShort(data[0]) catch return std.Io.File.Reader.Error.Unexpected;
    }

    pub fn fileReadPositional(userdata: ?*anyopaque, file: std.Io.File, data: []const []u8, offset: u64) std.Io.File.ReadPositionalError!usize {
        const self: *HF = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        return try self.performRead(handle, data, offset);
    }

    fn fileReadStreaming(userdata: ?*anyopaque, file: std.Io.File, data: []const []u8) std.Io.File.Reader.Error!usize {
        const self: *HF = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        const total = try self.performRead(handle, data, handle.pos);
        handle.pos += @intCast(total);
        return total;
    }

    fn fileClose(userdata: ?*anyopaque, files: []const std.Io.File) void {
        const self: *HF = @fieldParentPtr("base", VFSBase.as(userdata));
        for (files) |file| {
            self.closeHandle(@intCast(file.handle)) catch unreachable;
        }
    }
};

pub fn main() !void {
    const allocator = std.heap.smp_allocator;

    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();

    var client: std.http.Client = .{
        .allocator = allocator,
        .io = threaded.io(),
    };

    try client.initDefaultProxies(allocator);
    defer client.deinit();

    var hf: HF = try .init(allocator, threaded.io(), &client);
    defer hf.deinit();

    const io = hf.io();

    const file = std.Io.Dir.cwd().openFile(io, "Qwen/Qwen3-8B/model-00001-of-00005.safetensors", .{}) catch |err| {
        log.err("Failed to open HF file: {any}", .{err});
        unreachable;
    };

    log.info(">>>>>>> {any}", .{file});

    var reader = file.reader(io, &.{});
    reader.mode = reader.mode.toSimple();
    defer log.info("DONE", .{});

    // const data = try reader.interface.readAlloc(allocator, 1024);
    // defer allocator.free(data);
    // log.info("READ {s}", .{data});

    const buffer = try allocator.alloc(u8, 256 * 1024 * 1024);
    var stdout_writer = std.Io.File.stdout().writer(io, buffer);
    defer stdout_writer.interface.flush() catch {};

    _ = reader.interface.streamRemaining(&stdout_writer.interface) catch |err| {
        log.err("Failed to stream HF file: {any}", .{err});
        unreachable;
    };
}
