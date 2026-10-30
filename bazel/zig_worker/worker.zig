//! Bazel JSON persistent-worker protocol to Zig's compiler-server protocol.
const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const cwd = Io.Dir.cwd();
const outputName = "bazel_incremental";

pub const Request = struct {
    arguments: []const []const u8,
    inputs: []const Input = &.{},
    requestId: i32 = 0,
    verbosity: i32 = 0,

    pub const Input = struct { path: []const u8, digest: []const u8 = "" };
};

pub const Response = struct {
    exitCode: i32,
    output: []const u8,
    requestId: i32,
};

const Snapshot = struct {
    arena: *std.heap.ArenaAllocator,
    arguments: []const []const u8,
    inputs: std.StringHashMap(Input),

    const Input = struct {
        digest: []const u8,
        stat: ?struct { inode: Io.File.INode, size: u64, mtime: i96 },
    };

    fn init(gpa: Allocator, io: Io, request: Request) !Snapshot {
        const arena = try gpa.create(std.heap.ArenaAllocator);
        arena.* = .init(gpa);
        errdefer {
            arena.deinit();
            gpa.destroy(arena);
        }
        const alloc = arena.allocator();
        var args: std.ArrayList([]const u8) = .empty;
        for (request.arguments) |arg| {
            if (std.mem.startsWith(u8, arg, "@")) {
                const content = try cwd.readFileAlloc(io, arg[1..], alloc, .limited(16 * 1024 * 1024));
                var lines = std.mem.tokenizeScalar(u8, content, '\n');
                while (lines.next()) |line| try args.append(alloc, line);
            } else try args.append(alloc, try alloc.dupe(u8, arg));
        }
        var inputs = std.StringHashMap(Input).init(alloc);
        for (request.inputs) |input| {
            const stat = if (std.mem.endsWith(u8, input.path, ".zig")) try cwd.statFile(io, input.path, .{}) else null;
            try inputs.put(try alloc.dupe(u8, input.path), .{
                .digest = try alloc.dupe(u8, input.digest),
                .stat = if (stat) |s| .{ .inode = s.inode, .size = s.size, .mtime = s.mtime.nanoseconds } else null,
            });
        }
        return .{ .arena = arena, .arguments = try args.toOwnedSlice(alloc), .inputs = inputs };
    }

    fn deinit(self: Snapshot, gpa: Allocator) void {
        self.arena.deinit();
        gpa.destroy(self.arena);
    }

    fn reusable(self: Snapshot, previous: Snapshot) bool {
        if (self.arguments.len != previous.arguments.len or self.inputs.count() != previous.inputs.count()) return false;
        for (self.arguments, previous.arguments) |a, b| if (!std.mem.eql(u8, a, b)) return false;
        var it = self.inputs.iterator();
        while (it.next()) |entry| {
            const old = previous.inputs.get(entry.key_ptr.*) orelse return false;
            if (!std.mem.eql(u8, old.digest, entry.value_ptr.digest)) {
                // Zig checks source metadata. Restart for other input changes,
                // or source edits which preserve the metadata Zig checks.
                if (entry.value_ptr.stat == null or std.meta.eql(entry.value_ptr.stat, old.stat)) return false;
            }
        }
        return true;
    }
};

pub const Compiler = struct {
    arena: std.heap.ArenaAllocator,
    io: Io,
    child: std.process.Child,
    cache: []const u8,
    command: []const u8,
    outputs: []const Output,
    multiBuffer: Io.File.MultiReader.Buffer(2),
    multi: Io.File.MultiReader,
    updates: usize = 0,

    const Output = struct { kind: []const u8, path: []const u8 };

    fn init(gpa: Allocator, io: Io, zig: []const u8, arguments: []const []const u8) !*Compiler {
        const self = try gpa.create(Compiler);
        errdefer gpa.destroy(self);
        self.* = .{
            .arena = .init(gpa),
            .io = io,
            .child = undefined,
            .cache = undefined,
            .command = undefined,
            .outputs = undefined,
            .multiBuffer = undefined,
            .multi = undefined,
        };
        errdefer self.arena.deinit();
        const alloc = self.arena.allocator();
        var random: [16]u8 = undefined;
        io.random(&random);
        // A private cache avoids locks and mutable artifacts shared by sessions.
        self.cache = try std.fmt.allocPrint(alloc, "/tmp/zml-zig-worker-{s}", .{std.fmt.bytesToHex(random, .lower)});
        try cwd.createDir(io, self.cache, .fromMode(0o700));
        errdefer cwd.deleteTree(io, self.cache) catch {};
        self.command = try alloc.dupe(u8, arguments[0]);
        var args: std.ArrayList([]const u8) = .empty;
        var outputs: std.ArrayList(Output) = .empty;
        try args.append(alloc, zig);
        var skip = false;
        for (arguments) |arg| {
            if (skip) {
                skip = false;
                continue;
            }
            if (std.mem.eql(u8, arg, "--cache-dir") or std.mem.eql(u8, arg, "--name")) {
                skip = true;
            } else if (std.mem.startsWith(u8, arg, "-femit-") and std.mem.indexOfScalar(u8, arg, '=') != null) {
                const eq = std.mem.indexOfScalar(u8, arg, '=').?;
                try outputs.append(alloc, .{ .kind = try alloc.dupe(u8, arg[7..eq]), .path = try alloc.dupe(u8, arg[eq + 1 ..]) });
                try args.append(alloc, try alloc.dupe(u8, arg[0..eq]));
            } else if (std.mem.eql(u8, arg, "--watch") or std.mem.eql(u8, arg, "-fno-incremental") or std.mem.startsWith(u8, arg, "--listen")) {
                return error.ConflictingCompilerFlag;
            } else try args.append(alloc, arg);
        }
        try args.appendSlice(alloc, &.{ "--name", outputName, "--cache-dir", self.cache, "-fincremental", "--listen=-" });
        self.outputs = try outputs.toOwnedSlice(alloc);
        self.child = try std.process.spawn(io, .{ .argv = args.items, .stdin = .pipe, .stdout = .pipe, .stderr = .pipe });
        self.multi.init(gpa, io, self.multiBuffer.toStreams(), &.{ self.child.stdout.?, self.child.stderr.? });
        return self;
    }

    fn close(self: *Compiler, gpa: Allocator) void {
        self.multi.deinit();
        // Reap the compiler and close its pipes before removing mutable state.
        self.child.kill(self.io);
        cwd.deleteTree(self.io, self.cache) catch {};
        self.arena.deinit();
        gpa.destroy(self);
    }

    fn update(self: *Compiler, gpa: Allocator, diagnostics: *Io.Writer) !i32 {
        var stdinBuffer: [8]u8 = undefined;
        var stdin = self.child.stdin.?.writerStreaming(self.io, &stdinBuffer);
        var client: std.zig.Client = .{ .in = self.multi.reader(0), .out = &stdin.interface };
        try client.serveBodylessMessage(.update);
        var digest: ?[std.Build.Cache.bin_digest_len]u8 = null;
        while (true) {
            const header = client.receiveMessageWithMultiReader(&self.multi, .none) catch |err| {
                try diagnostics.writeAll(self.multi.reader(1).buffered());
                return err;
            };
            const body = try client.in.take(header.bytes_len);
            switch (header.tag) {
                .zig_version => if (!std.mem.eql(u8, body, builtin.zig_version_string)) {
                    try diagnostics.print("compiler protocol version mismatch: worker {s}, compiler {s}\n", .{ builtin.zig_version_string, body });
                    return error.CompilerVersionMismatch;
                },
                .emit_digest => {
                    if (body.len != 1 + std.Build.Cache.bin_digest_len) return error.InvalidDigest;
                    digest = body[1..][0..std.Build.Cache.bin_digest_len].*;
                },
                .error_bundle => {
                    var errors = try std.zig.Server.allocErrorBundle(gpa, body);
                    defer errors.deinit(gpa);
                    self.updates += 1;
                    try diagnostics.writeAll(self.multi.reader(1).buffered());
                    self.multi.reader(1).toss(self.multi.reader(1).bufferedLen());
                    if (errors.errorMessageCount() > 0) {
                        try errors.renderToWriter(.{}, diagnostics);
                        return 1;
                    }
                    if (self.outputs.len > 0) try self.copyOutputs(digest orelse return error.MissingDigest);
                    return 0;
                },
                else => {},
            }
        }
    }

    fn copyOutputs(self: *Compiler, digest: [std.Build.Cache.bin_digest_len]u8) !void {
        var arena: std.heap.ArenaAllocator = .init(self.arena.child_allocator);
        defer arena.deinit();
        const alloc = arena.allocator();
        const directory = try std.fmt.allocPrint(alloc, "{s}/o/{s}", .{ self.cache, std.fmt.bytesToHex(digest, .lower) });
        for (self.outputs) |output| {
            const names: []const []const u8 = if (std.mem.eql(u8, output.kind, "bin")) blk: {
                if (std.mem.eql(u8, self.command, "build-lib")) break :blk &.{ "lib" ++ outputName ++ ".a", outputName ++ ".lib", "lib" ++ outputName ++ ".so", "lib" ++ outputName ++ ".dylib", outputName ++ ".dll" };
                if (std.mem.eql(u8, self.command, "build-obj") or std.mem.eql(u8, self.command, "test-obj")) break :blk &.{ outputName ++ ".o", outputName ++ ".obj" };
                break :blk &.{ outputName, outputName ++ ".exe" };
            } else if (std.mem.eql(u8, output.kind, "asm")) &.{outputName ++ ".s"} else if (std.mem.eql(u8, output.kind, "llvm-ir")) &.{outputName ++ ".ll"} else if (std.mem.eql(u8, output.kind, "llvm-bc")) &.{outputName ++ ".bc"} else if (std.mem.eql(u8, output.kind, "h")) &.{outputName ++ ".h"} else return error.UnsupportedOutput;
            var source: ?[]const u8 = null;
            for (names) |name| {
                const path = try std.fs.path.join(alloc, &.{ directory, name });
                _ = cwd.statFile(self.io, path, .{}) catch |err| switch (err) {
                    error.FileNotFound => continue,
                    else => return err,
                };
                if (source != null) return error.AmbiguousOutput;
                source = path;
            }
            // The compiler modifies its cache in place; Bazel must own a copy.
            try cwd.copyFile(source orelse return error.MissingOutput, cwd, output.path, self.io, .{ .make_path = true, .replace = true });
        }
    }
};

pub const Worker = struct {
    gpa: Allocator,
    io: Io,
    zig: []const u8,
    compiler: ?*Compiler = null,
    previous: ?Snapshot = null,

    pub fn deinit(self: *Worker) void {
        self.stopCompiler();
        if (self.previous) |previous| previous.deinit(self.gpa);
    }

    fn stopCompiler(self: *Worker) void {
        if (self.compiler) |compiler| compiler.close(self.gpa);
        self.compiler = null;
    }

    pub fn build(self: *Worker, request: Request, output: *Io.Writer) !i32 {
        const snapshot = try Snapshot.init(self.gpa, self.io, request);
        var transferred = false;
        errdefer if (!transferred) snapshot.deinit(self.gpa);
        const restart = self.compiler == null or self.previous == null or !snapshot.reusable(self.previous.?);
        if (restart) {
            self.stopCompiler();
            self.compiler = try Compiler.init(self.gpa, self.io, self.zig, snapshot.arguments);
        }
        if (self.previous) |previous| previous.deinit(self.gpa);
        self.previous = snapshot;
        transferred = true;
        return self.compile(request, restart, output);
    }

    fn compile(self: *Worker, request: Request, restart: bool, output: *Io.Writer) !i32 {
        const compiler = self.compiler.?;
        errdefer self.stopCompiler();
        const code = try compiler.update(self.gpa, output);
        if (request.verbosity > 0) try output.print("zig worker: pid={d} update={d} restarted={}\n", .{ compiler.child.id.?, compiler.updates, restart });
        return code;
    }
};

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var zig: ?[]const u8 = null;
    var persistent = false;
    var arguments: std.ArrayList([]const u8) = .empty;
    for (args[1..]) |arg| {
        if (std.mem.cutPrefix(u8, arg, "--zig=")) |path| zig = path else if (std.mem.startsWith(u8, arg, "--action=")) continue else if (std.mem.eql(u8, arg, "--persistent_worker")) persistent = true else try arguments.append(init.arena.allocator(), arg);
    }
    var worker: Worker = .{ .gpa = init.gpa, .io = init.io, .zig = zig orelse return error.MissingCompiler };
    defer worker.deinit();
    var stdoutBuffer: [4096]u8 = undefined;
    var stdout = Io.File.stdout().writerStreaming(init.io, &stdoutBuffer);
    if (!persistent) {
        var stderrBuffer: [4096]u8 = undefined;
        var stderr = Io.File.stderr().writerStreaming(init.io, &stderrBuffer);
        const code = try worker.build(.{ .arguments = arguments.items }, &stderr.interface);
        try stderr.interface.flush();
        return @intCast(code);
    }
    var stdinBuffer: [8192]u8 = undefined;
    var stdin = Io.File.stdin().readerStreaming(init.io, &stdinBuffer);
    while (true) {
        var arena: std.heap.ArenaAllocator = .init(init.gpa);
        defer arena.deinit();
        const alloc = arena.allocator();
        var line: Io.Writer.Allocating = .init(alloc);
        _ = stdin.interface.streamDelimiter(&line.writer, '\n') catch |err| switch (err) {
            error.EndOfStream => break,
            else => return err,
        };
        stdin.interface.toss(1);
        const parsed = try std.json.parseFromSlice(Request, alloc, line.written(), .{ .ignore_unknown_fields = true });
        var diagnostics: Io.Writer.Allocating = .init(alloc);
        const code = worker.build(parsed.value, &diagnostics.writer) catch |err| blk: {
            try diagnostics.writer.print("zig worker: {t}\n", .{err});
            break :blk 1;
        };
        try std.json.Stringify.value(Response{ .exitCode = code, .output = diagnostics.written(), .requestId = parsed.value.requestId }, .{}, &stdout.interface);
        try stdout.interface.writeByte('\n');
        try stdout.interface.flush();
    }
    return 0;
}
