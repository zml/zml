const builtin = @import("builtin");
const c = @import("c");
const mlir = @import("mlir");
const runfiles = @import("runfiles");
const runtimes = @import("runtimes");
const std = @import("std");
const stdx = @import("stdx");

const platform = @import("platform.zig");
const pjrt = @import("pjrtx.zig");

const HostBuffer = @import("hostbuffer.zig").HostBuffer;
const PjrtApiMap = std.EnumArray(Target, ?*const pjrt.Api);
const Platform = @import("platform.zig").Platform;
const PlatformsMap = std.EnumArray(Target, ?Platform);
const Target = @import("platform.zig").Target;

const available_targets = @import("platform.zig").available_targets;
const log = std.log.scoped(.@"zml/context");

/// Every program using ZML must start with a `zml.Context.init(.{});`
/// The ZML context contains global state to interact with the different
/// devices available on your system.
/// Note that the runtimes available depends on how the program was compiled.
/// For example you need to compile your program with `--//runtimes:cuda=true`
/// to have the CUDA runtime available.
pub const Context = struct {
    var apis = PjrtApiMap.initFill(null);
    var apis_once = std.once(struct {
        fn call() void {
            inline for (comptime std.enums.values(runtimes.Platform)) |t| {
                if (runtimes.load(t)) |api| {
                    Context.apis.set(t, api);
                } else |_| {}
            }
        }
    }.call);

    var mlir_once = std.once(struct {
        fn call() void {
            mlir.registerPasses("Transforms");
        }
    }.call);

    var runfiles_once = std.once(struct {
        fn call_() !void {
            if (std.process.hasEnvVarConstant("RUNFILES_MANIFEST_FILE") or std.process.hasEnvVarConstant("RUNFILES_DIR")) {
                return;
            }

            var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
            const allocator = arena.allocator();
            defer arena.deinit();

            var envMap = std.process.EnvMap.init(allocator);
            var r = (try runfiles.Runfiles.create(.{ .allocator = allocator })) orelse return;
            try r.environment(&envMap);

            var it = envMap.iterator();
            while (it.next()) |entry| {
                const keyZ = try allocator.dupeZ(u8, entry.key_ptr.*);
                const valueZ = try allocator.dupeZ(u8, entry.value_ptr.*);
                _ = c.setenv(keyZ.ptr, valueZ.ptr, 1);
            }
        }

        fn call() void {
            call_() catch @panic("Unable to init runfiles env");
        }
    }.call);

    platforms: PlatformsMap,

    /// Creates a ZML Context and returns it.
    pub fn init() !Context {
        Context.runfiles_once.call();
        Context.apis_once.call();
        Context.mlir_once.call();

        var platforms = PlatformsMap.initFill(null);
        var num_platforms: u8 = 0;
        var it = Context.apis.iterator();
        while (it.next()) |entry| {
            if (entry.value.*) |api| {
                const target = entry.key;
                const p = Platform.init(target, api, .{}) catch |err| {
                    log.err("Failed to load platform .{s}: {}", .{ @tagName(target), err });
                    continue;
                };
                if (p.getDevices().len == 0) {
                    log.err("No device found for platform {} !", .{target});
                    continue;
                }
                if (target == .cuda) {
                    try cuda.registerZmlCustomCalls(p);
                }
                platforms.set(target, p);
                num_platforms += 1;
            }
        }
        if (num_platforms == 0) {
            log.err("No platform available", .{});
            return error.NoPlatformAvailable;
        }

        return .{
            .platforms = platforms,
        };
    }

    fn platformToLibrary(comptime target: Target) []const u8 {
        const ext = switch (builtin.os.tag) {
            .windows => ".dll",
            .macos, .ios, .watchos => ".dylib",
            else => ".so",
        };
        return switch (target) {
            inline else => "libpjrt_" ++ @tagName(target) ++ ext,
        };
    }

    pub fn pjrtApi(target: Target) *const pjrt.Api {
        return Context.apis.get(target).?;
    }

    pub fn deinit(self: *Context) void {
        var iterator = self.platforms.iterator();
        while (iterator.next()) |entry| {
            if (entry.value.*) |*p| {
                p.deinit();
            }
        }
        self.* = undefined;
    }

    /// Automatically selects the best Platform loaded in the current Context.
    ///
    /// For example, if supported, this will select a platform corresponding to an accelerator (GPU, TPU, ...).
    pub fn autoPlatform(self: *Context) Platform {
        // the last platform is the one that with the high enum number, so considered
        // to be the "best" one
        var platform_: ?Platform = null;
        var iterator = self.platforms.iterator();
        while (iterator.next()) |entry| {
            if (entry.value.*) |p| {
                platform_ = p;
            }
        }
        return platform_ orelse @panic("No platform found !");
    }

    pub fn printAvailablePlatforms(self: Context, selected: platform.Platform) void {
        // List available targets
        log.info("Available Platforms:", .{});
        const selected_prefix = "✅";
        const not_selected_prefix = "• ";
        const selected_postfix = "(AUTO-SELECTED)";
        const not_selected_postfix = "";

        for (platform.available_targets) |target| {
            log.info("  {s} {s} {s}", .{
                if (target == selected.target) selected_prefix else not_selected_prefix,
                @tagName(target),
                if (target == selected.target) selected_postfix else not_selected_postfix,
            });

            // now the platform's devices
            if (self.platforms.get(target)) |pfm| {
                for (pfm.getDevices(), 0..) |device, index| {
                    const deviceKind = device.getDescription(pfm.pjrt_api).getKind(pfm.pjrt_api);
                    log.info("       ◦ #{d}: {s}", .{
                        index,
                        deviceKind,
                    });
                    // we only list 1 CPU device
                    if (target == .cpu) break;
                }
            }
        }
    }

    pub const HostCallbackCtx = struct {
        host: HostBuffer,
        mutex: std.Thread.Mutex = std.Thread.Mutex{},
    };
    pub const HostCallback = fn (HostBuffer) void;
};

const cuda = struct {
    var runtime: Runtime = undefined;

    pub fn registerZmlCustomCalls(cuda_platform: Platform) !void {
        std.debug.assert(cuda_platform.target == .cuda);

        cuda.runtime = try Runtime.init();
        const registry = cuda_platform.pjrt_api.customCallRegistry().?;
        try registry.register(cuda_platform.pjrt_api, 0, "zmlHostBufferCallback", &hostBufferCallback);
    }

    pub const Stream = opaque {};
    pub const MemcpyKind = enum(c_int) {
        host_to_host = 0,
        host_to_device = 1,
        device_to_host = 2,
        device_to_device = 3,
        default = 4,
    };

    pub const Runtime = struct {
        memcpyAsync: MemcpyAsync,
        streamSynchronize: StreamSynchronize,

        const MemcpyAsync = *const fn (dst: *anyopaque, src: *const anyopaque, count: usize, kind: MemcpyKind, stream: *Stream) callconv(.C) c_int;
        const StreamSynchronize = *const fn (stream: *Stream) callconv(.C) c_int;

        pub fn init() !Runtime {
            var cudart = try std.DynLib.open("libcudart.so.12");
            defer cudart.close();

            return .{
                .memcpyAsync = cudart.lookup(Runtime.MemcpyAsync, "cudaMemcpyAsync") orelse return error.NotFound,
                .streamSynchronize = cudart.lookup(Runtime.StreamSynchronize, "cudaStreamSynchronize") orelse return error.NotFound,
            };
        }
    };

    fn getContext(args: [*]const u8, args_len: usize) struct { *const Context.HostCallback, *Context.HostCallbackCtx } {
        std.debug.assert(args_len == @sizeOf(*anyopaque) * 2);

        const raw_fn_ptr: usize = @bitCast(args[0..@sizeOf(*anyopaque)].*);
        const fn_ptr: *const Context.HostCallback = @ptrFromInt(raw_fn_ptr);

        const raw_ctx_ptr: usize = @bitCast(args[@sizeOf(*anyopaque)..][0..@sizeOf(*anyopaque)].*);
        const ctx_ptr: *Context.HostCallbackCtx = @ptrFromInt(raw_ctx_ptr);
        return .{ fn_ptr, ctx_ptr };
    }

    fn hostBufferCallback(opaque_stream: *anyopaque, buffers: [*]*anyopaque, args: [*]const u8, args_len: usize) callconv(.C) void {
        const stream: *Stream = @ptrCast(opaque_stream);
        const src: *anyopaque = buffers[0];
        const callback, const ctx = getContext(args, args_len);

        // Add synchronization because this is called from the device driver.
        ctx.mutex.lock();
        defer ctx.mutex.unlock();

        const host_dst: []u8 = @constCast(ctx.host.data);
        const memcpy_result = cuda.runtime.memcpyAsync(host_dst.ptr, src, host_dst.len, .device_to_host, stream);
        _ = memcpy_result;
        const synchronize_result = cuda.runtime.streamSynchronize(stream);
        _ = synchronize_result;

        callback(ctx.host);
    }
};
