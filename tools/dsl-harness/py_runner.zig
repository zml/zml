//! Long-lived subprocess wrapper around a Bazel-built `py_binary` runner.
//! Spawn one per kernel up-front so the Triton/JAX import cost is paid
//! once, push every sweep through the same child, then `deinit()`.
//!
//! Wire protocol (one JSON request + one response per line):
//!
//!   request  : {"cfg": {...}}
//!   response : {"ok": true,  "ttir": "..."}      // triton kernel
//!            | {"ok": true,  "mosaic": "..."}    // mosaic kernel
//!            | {"ok": false, "error": "..."}

const std = @import("std");

const Allocator = std.mem.Allocator;
const Io = std.Io;

const log = std.log.scoped(.@"harness/py_runner");

pub const Response = struct {
    ttir: ?[]const u8 = null,
    mosaic: ?[]const u8 = null,
};

pub const Runner = struct {
    allocator: Allocator,
    io: Io,
    child: std.process.Child,

    stdin_buf: [4096]u8,
    stdin_writer: Io.File.Writer,

    stdout_buf: []u8,
    stdout_reader: Io.File.Reader,

    pub fn spawn(
        allocator: Allocator,
        io: Io,
        exe_path: []const u8,
        extra_args: []const []const u8,
        environ_map: *const std.process.Environ.Map,
    ) !*Runner {
        const argv = try allocator.alloc([]const u8, 1 + extra_args.len);
        defer allocator.free(argv);
        argv[0] = exe_path;
        for (extra_args, 0..) |a, i| argv[1 + i] = a;

        const child = try std.process.spawn(io, .{
            .argv = argv,
            .environ_map = environ_map,
            .stdin = .pipe,
            .stdout = .pipe,
            .stderr = .inherit,
        });

        const self = try allocator.create(Runner);
        self.* = .{
            .allocator = allocator,
            .io = io,
            .child = child,
            .stdin_buf = undefined,
            .stdin_writer = undefined,
            .stdout_buf = try allocator.alloc(u8, 1 << 20),
            .stdout_reader = undefined,
        };
        if (child.stdin) |pipe| self.stdin_writer = pipe.writer(io, &self.stdin_buf);
        if (child.stdout) |pipe| self.stdout_reader = pipe.reader(io, self.stdout_buf);
        return self;
    }

    pub fn deinit(self: *Runner) void {
        if (self.child.stdin) |stdin| {
            stdin.close(self.io);
            self.child.stdin = null;
        }
        _ = self.child.wait(self.io) catch |err| {
            log.warn("child wait failed: {s}", .{@errorName(err)});
        };
        self.allocator.free(self.stdout_buf);
        self.allocator.destroy(self);
    }

    /// `cfg_json` is verbatim JSON from `cfgJsonFn`; it is trusted
    /// harness-emitted JSON, so it is spliced in without re-escaping.
    pub fn requestCompile(self: *Runner, arena: Allocator, cfg_json: []const u8) !Response {
        if (self.child.stdin == null) return error.RunnerExited;
        const w = &self.stdin_writer.interface;
        try w.print("{{\"cfg\":{s}}}\n", .{cfg_json});
        try w.flush();

        const line = try self.readResponseLine();

        const ResponseJson = struct {
            ok: bool,
            @"error": ?[]const u8 = null,
            ttir: ?[]const u8 = null,
            mosaic: ?[]const u8 = null,
        };
        const r = std.json.parseFromSliceLeaky(
            ResponseJson,
            arena,
            line,
            .{ .ignore_unknown_fields = true },
        ) catch |err| {
            log.err("py_runner: failed to parse '{s}': {s}", .{ line, @errorName(err) });
            return error.ProtocolError;
        };
        if (!r.ok) {
            log.err("py_runner: child reports failure: {s}", .{r.@"error" orelse "unknown"});
            return error.ProtocolError;
        }
        return .{
            .ttir = r.ttir,
            .mosaic = r.mosaic,
        };
    }

    /// Inclusive variant — the exclusive form leaves the `\n` at seek=0
    /// and the next call returns 0 bytes.
    fn readResponseLine(self: *Runner) ![]const u8 {
        const line_with_nl = self.stdout_reader.interface.takeDelimiterInclusive('\n') catch |err| {
            log.err("py_runner: stdout read failed: {s}", .{@errorName(err)});
            return error.RunnerExited;
        };
        return std.mem.trimEnd(u8, line_with_nl, "\n");
    }
};
