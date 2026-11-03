const std = @import("std");
const adapter = @import("worker.zig");
const testing = std.testing;
const io = testing.io;
const cwd = std.Io.Dir.cwd();

const Fixture = struct {
    arena: *std.heap.ArenaAllocator,
    tmp: testing.TmpDir,
    root: []const u8,
    output: []const u8,
    worker: adapter.Worker,

    fn init() !Fixture {
        const arena = try testing.allocator.create(std.heap.ArenaAllocator);
        arena.* = .init(testing.allocator);
        errdefer {
            arena.deinit();
            testing.allocator.destroy(arena);
        }
        const alloc = arena.allocator();
        var tmp = testing.tmpDir(.{});
        errdefer tmp.cleanup();
        const root = try tmp.dir.realPathFileAlloc(io, ".", alloc);
        var self: Fixture = .{
            .arena = arena,
            .tmp = tmp,
            .root = root,
            .output = try std.fs.path.join(alloc, &.{ root, "program" }),
            .worker = .{
                .gpa = testing.allocator,
                .io = io,
                .zig = try testing.environ.getAlloc(alloc, "ZIG_WORKER_TEST_COMPILER"),
            },
        };
        try self.write("main.zig", "pub fn main() u8 { return @import(\"value.zig\").value; }\n");
        try self.write("value.zig", "pub const value: u8 = 42;\n");
        try self.write("value.txt", "x");
        return self;
    }

    fn deinit(self: *Fixture) void {
        self.worker.deinit();
        self.tmp.cleanup();
        self.arena.deinit();
        testing.allocator.destroy(self.arena);
    }

    fn write(self: *Fixture, path: []const u8, contents: []const u8) !void {
        try self.tmp.dir.writeFile(io, .{ .sub_path = path, .data = contents });
    }

    fn request(self: *Fixture, extra: []const []const u8) !adapter.Request {
        const alloc = self.arena.allocator();
        var args: std.ArrayList([]const u8) = .empty;
        try args.appendSlice(alloc, &.{
            "build-exe",                                                     try std.fs.path.join(alloc, &.{ self.root, "main.zig" }),
            try std.fmt.allocPrint(alloc, "-femit-bin={s}", .{self.output}), "--global-cache-dir",
            try std.fs.path.join(alloc, &.{ self.root, "global-cache" }),    "-fno-llvm",
            "-fno-lld",                                                      "-fnew-linker",
        });
        try args.appendSlice(alloc, extra);
        const inputs = try alloc.alloc(adapter.Request.Input, 3);
        for (inputs, [_][]const u8{ "main.zig", "value.zig", "value.txt" }) |*input, name| {
            const data = try self.tmp.dir.readFileAlloc(io, name, alloc, .limited(1024));
            var digest: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(data, &digest, .{});
            input.* = .{
                .path = try std.fs.path.join(alloc, &.{ self.root, name }),
                .digest = try alloc.dupe(u8, &std.fmt.bytesToHex(digest, .lower)),
            };
        }
        return .{ .arguments = try args.toOwnedSlice(alloc), .inputs = inputs };
    }

    fn run(self: *Fixture, expected: u8) !void {
        const result = try std.process.run(self.arena.allocator(), io, .{ .argv = &.{self.output} });
        try testing.expectEqual(std.process.Child.Term{ .exited = expected }, result.term);
    }

    fn build(self: *Fixture, expected: u8, extra: []const []const u8) !std.process.Child.Id {
        var diagnostics: std.Io.Writer.Allocating = .init(self.arena.allocator());
        const code = try self.worker.build(try self.request(extra), &diagnostics.writer);
        if (code != 0) std.debug.print("{s}\n", .{diagnostics.written()});
        try testing.expectEqual(0, code);
        try self.run(expected);
        return self.worker.compiler.?.child.id.?;
    }
};

test "reuse imports, recreate outputs, and recover from compiler diagnostics" {
    var f = try Fixture.init();
    defer f.deinit();
    const pid = try f.build(42, &.{});
    try f.write("value.zig", "pub const value: u8 = 43;\n");
    try cwd.deleteFile(io, f.output);
    try testing.expectEqual(pid, try f.build(43, &.{}));
    try cwd.deleteFile(io, f.output);
    try testing.expectEqual(pid, try f.build(43, &.{}));
    try f.write("value.zig", "pub const value: u8 = @compileError(\"intentional failure\");\n");
    try cwd.deleteFile(io, f.output);
    var diagnostics: std.Io.Writer.Allocating = .init(f.arena.allocator());
    try testing.expectEqual(1, try f.worker.build(try f.request(&.{}), &diagnostics.writer));
    try testing.expect(std.mem.indexOf(u8, diagnostics.written(), "intentional failure") != null);
    try testing.expect(std.mem.indexOf(u8, diagnostics.written(), "value.zig:1:") != null);
    try testing.expectError(error.FileNotFound, cwd.statFile(io, f.output, .{}));
    try f.write("value.zig", "pub const value: u8 = 44;\n");
    try testing.expectEqual(pid, try f.build(44, &.{}));
}

test "restart for changed non-source inputs, arguments, or input membership" {
    var f = try Fixture.init();
    defer f.deinit();
    var pid = try f.build(42, &.{});
    try f.write("value.txt", "changed embedded input");
    const afterInput = try f.build(42, &.{});
    try testing.expect(afterInput != pid);
    pid = try f.build(42, &.{"-OReleaseSafe"});
    try testing.expect(pid != afterInput);
    var request = try f.request(&.{"-OReleaseSafe"});
    request.inputs = request.inputs[0..2];
    var diagnostics: std.Io.Writer.Allocating = .init(f.arena.allocator());
    try testing.expectEqual(0, try f.worker.build(request, &diagnostics.writer));
    try testing.expect(f.worker.compiler.?.child.id.? != pid);
    try f.run(42);
}

test "restart when a source edit preserves file metadata" {
    var f = try Fixture.init();
    defer f.deinit();
    const pid = try f.build(42, &.{});
    const stat = try f.tmp.dir.statFile(io, "value.zig", .{});
    try f.write("value.zig", "pub const value: u8 = 43;\n");
    try f.tmp.dir.setTimestamps(io, "value.zig", .{ .modify_timestamp = .{ .new = stat.mtime } });
    try testing.expect(try f.build(43, &.{}) != pid);
}

test "compiler exit discards the session and the next request recovers" {
    var f = try Fixture.init();
    defer f.deinit();
    const pid = try f.build(42, &.{});
    try std.posix.kill(pid, .KILL);
    var diagnostics: std.Io.Writer.Allocating = .init(f.arena.allocator());
    const request = try f.request(&.{});
    if (f.worker.build(request, &diagnostics.writer)) |_| return error.ExpectedCompilerFailure else |_| {}
    try testing.expect(f.worker.compiler == null);
    try testing.expect(try f.build(42, &.{}) != pid);
}

test "JSON request IDs, parameter files, reuse, and EOF shutdown" {
    var f = try Fixture.init();
    defer f.deinit();
    const alloc = f.arena.allocator();
    const binary = try testing.environ.getAlloc(alloc, "ZIG_WORKER_TEST_BINARY");
    var child = try std.process.spawn(io, .{
        .argv = &.{ binary, try std.fmt.allocPrint(alloc, "--zig={s}", .{f.worker.zig}), "--action=test", "--persistent_worker" },
        .stdin = .pipe,
        .stdout = .pipe,
        .stderr = .inherit,
    });
    defer child.kill(io);
    var writeBuffer: [4096]u8 = undefined;
    var stdin = child.stdin.?.writerStreaming(io, &writeBuffer);
    var readBuffer: [4096]u8 = undefined;
    var stdout = child.stdout.?.readerStreaming(io, &readBuffer);
    var previousPid: ?[]const u8 = null;
    for (0..2) |i| {
        try f.write("value.zig", try std.fmt.allocPrint(alloc, "pub const value: u8 = {d};\n", .{42 + i}));
        var request = try f.request(&.{});
        const params = try std.mem.join(alloc, "\n", request.arguments);
        try f.write("args.params", params);
        request.arguments = &.{try std.fmt.allocPrint(alloc, "@{s}/args.params", .{f.root})};
        request.requestId = @intCast(i + 7);
        request.verbosity = 1;
        try std.json.Stringify.value(request, .{}, &stdin.interface);
        try stdin.interface.writeByte('\n');
        try stdin.interface.flush();
        const line = try stdout.interface.takeDelimiterInclusive('\n');
        const response = try std.json.parseFromSlice(adapter.Response, alloc, line, .{});
        try testing.expectEqual(request.requestId, response.value.requestId);
        if (response.value.exitCode != 0) std.debug.print("{s}\n", .{response.value.output});
        try testing.expectEqual(0, response.value.exitCode);
        const pidStart = (std.mem.indexOf(u8, response.value.output, "pid=") orelse return error.MissingPid) + 4;
        const pid = std.mem.sliceTo(response.value.output[pidStart..], ' ');
        if (previousPid) |previous| try testing.expectEqualStrings(previous, pid);
        previousPid = try alloc.dupe(u8, pid);
        try f.run(@intCast(42 + i));
    }
    child.stdin.?.close(io);
    child.stdin = null;
    try testing.expect((try child.wait(io)).success());
}
