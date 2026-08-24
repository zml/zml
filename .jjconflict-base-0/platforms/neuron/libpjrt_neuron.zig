const std = @import("std");

const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const neuron = @import("neuron.zig");
const runfiles = @import("runfiles");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/platforms/neuron");

fn findFreeTcpPort(io: std.Io) !u16 {
    const addr: std.Io.net.IpAddress = .{ .ip4 = .{ .bytes = .{ 127, 0, 0, 1 }, .port = 0 } };
    var server = try std.Io.net.IpAddress.listen(
        &addr,
        io,
        .{},
    );
    defer server.deinit(io);
    return server.socket.address.getPort();
}

pub export fn zmlxneuron_dlopen(filename: [*c]const u8, flags: c_int) ?*anyopaque {
    const replacements: std.StaticStringMap([:0]const u8) = .initComptime(.{
        .{ "libnccom.so", "libnccom.so.2" },
        .{ "libnrt.so", "libnrt.so.1" },
        .{ "libncfw.so", "libncfw.so.2" },
    });

    var buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const new_filename: [*c]const u8 = if (filename) |f| blk: {
        const replacement = replacements.get(std.Io.Dir.path.basename(std.mem.span(f))) orelse break :blk f;
        break :blk stdx.Io.Dir.path.bufJoinZ(&buf, &.{
            stdx.process.selfSharedObjectDirPath(),
            replacement,
        }) catch unreachable;
    } else null;

    return std.c.dlopen(new_filename, @bitCast(flags));
}

extern fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

fn setupNeuronEnv(io: std.Io, sandbox_path: []const u8) !void {
    const scratchpad_page_size = "1024";

    var root_comm_buf: [256]u8 = undefined;
    _ = setenv(
        "NEURON_RT_ROOT_COMM_ID",
        try std.fmt.bufPrintZ(&root_comm_buf, "127.0.0.1:{d}", .{try findFreeTcpPort(io)}),
        1,
    );
    _ = setenv(
        "NEURON_INTERNAL_PJRT_C_API_VERSION",
        std.fmt.comptimePrint("{d}.{d}", .{
            c.PJRT_API_MAJOR,
            c.PJRT_API_MINOR,
        }),
        1,
    );
    _ = setenv("NEURON_RT_LOG_LEVEL", "error", 0);
    _ = setenv("NEURON_RT_DISABLE_EXECUTION_BARRIER", "1", 0);
    _ = setenv("NEURON_RT_STOCHASTIC_ROUNDING_EN", "1", 1);
    _ = setenv("NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS", "3", 0);
    _ = setenv("NEURON_SCRATCHPAD_PAGE_SIZE", scratchpad_page_size, 0);

    var bin_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const bin_path = try stdx.Io.Dir.path.bufJoin(&bin_path_buf, &.{ sandbox_path, "bin" });

    const old_path = if (std.c.getenv("PATH")) |path| std.mem.span(path) else "";
    var new_path_buf: [std.posix.PATH_MAX:0]u8 = undefined;
    const new_path = if (old_path.len == 0)
        try std.fmt.bufPrintZ(&new_path_buf, "{s}", .{bin_path})
    else
        try std.fmt.bufPrintZ(&new_path_buf, "{s}:{s}", .{ bin_path, old_path });
    if (setenv("PATH", new_path.ptr, 1) != 0) return error.SetEnvFailed;

    if (std.c.getenv("NEURON_CC_FLAGS") != null) return;

    const log_level = if (std.c.getenv("NEURON_RT_LOG_LEVEL")) |level| std.mem.span(level) else "error";

    var flags_buf: [512:0]u8 = undefined;
    const instance = try neuron.instance();
    const target = @tagName(instance.compilerTarget());
    const flags = try std.fmt.bufPrintZ(
        &flags_buf,
        "--target={s} " ++
            "--optlevel=1 " ++
            "--model-type=transformer " ++
            "--enable-fast-loading-neuron-binaries " ++
            "--enable-fast-context-switch " ++
            "--hbm-scratchpad-page-size={s} " ++
            "--verbose {s} " ++
            "--logfile-verbose {s} " ++
            "--logfile=./log-neuron-cc.txt",
        .{
            target,
            scratchpad_page_size,
            log_level,
            log_level,
        },
    );

    _ = setenv("NEURON_CC_FLAGS", flags.ptr, 0);
}

// Duplicates a PJRT Api object while being careful about struct size differences
fn dupePjrtApi(api: *c.PJRT_Api) c.PJRT_Api {
    var ret: c.PJRT_Api = undefined;
    const struct_size = @min(@sizeOf(c.PJRT_Api), api.struct_size);
    @memcpy(
        std.mem.asBytes(&ret)[0..struct_size],
        std.mem.asBytes(api)[0..struct_size],
    );
    return ret;
}

fn getPjrtApi() !*c.PJRT_Api {
    var threaded: std.Io.Threaded = .init(std.heap.c_allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const Static = struct {
        var inner: *c.PJRT_Api = undefined;
        var proxy: c.PJRT_Api = undefined;
    };

    var sandbox_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox_path = try stdx.Io.Dir.path.bufJoin(&sandbox_path_buf, &.{
        stdx.process.selfSharedObjectDirPath(),
        "..",
    });

    try setupNeuronEnv(io, sandbox_path);

    Static.inner = blk: {
        const GetPjrtApi_inner = GetPjrtApi_blk: {
            var lib: std.DynLib = .{
                .inner = .{
                    .handle = handle_blk: {
                        var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
                        const library = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libneuronpjrt.so" });
                        break :handle_blk std.c.dlopen(library, .{ .LAZY = true, .NODELETE = true }) orelse {
                            log.err("Unable to dlopen plugin: {?s}", .{std.mem.span(std.c.dlerror())});
                            return error.DlOpenFailed;
                        };
                    },
                },
            };

            break :GetPjrtApi_blk lib.lookup(*const fn () callconv(.c) *c.PJRT_Api, "GetPjrtApi") orelse {
                log.err("Unable to find symbol GetPjrtApi in plugin: {?s}", .{std.mem.span(std.c.dlerror())});
                return error.SymbolNotFound;
            };
        };

        break :blk GetPjrtApi_inner();
    };

    Static.proxy = dupePjrtApi(Static.inner);
    // Setup the API proxy functions
    // Static.proxy.PJRT_Plugin_Attributes = &struct {
    //     const STRUCT_SIZE = 24; // according to the failing assertion

    //     fn call(args: [*c]c.PJRT_Plugin_Attributes_Args) callconv(.c) ?*c.PJRT_Error {
    //         var new_args = args.*;
    //         new_args.struct_size = @min(new_args.struct_size, STRUCT_SIZE);
    //         return Static.inner.PJRT_Plugin_Attributes.?(&new_args);
    //     }
    // }.call;

    return &Static.proxy;
}

pub export fn GetPjrtApi() ?*c.PJRT_Api {
    return getPjrtApi() catch |err| {
        log.err("Failed to get PJRT API: {}", .{err});
        return null;
    };
}
