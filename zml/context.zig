const std = @import("std");
const builtin = @import("builtin");

const c = @import("c");
const mlir = @import("mlir");
const runfiles = @import("runfiles");
const runtimes = @import("runtimes");
const stdx = @import("stdx");

const DataType = @import("dtype.zig").DataType;
const HostBuffer = @import("hostbuffer.zig").HostBuffer;
const pjrt = @import("pjrtx.zig");
const Platform = @import("platform.zig").Platform;
const Shape = @import("shape.zig").Shape;
const Target = @import("platform.zig").Target;
const zml_platform = @import("platform.zig");

const PjrtApiMap = std.EnumArray(Target, ?*const pjrt.Api);
const PlatformsMap = std.EnumArray(Target, ?Platform);
const log = std.log.scoped(.@"zml/context");

test {
    std.testing.refAllDecls(Context);
}

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
                    if (t == .cuda) cuda.init();
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

        var num_platforms: u8 = 0;
        for (Context.apis.values) |api| {
            if (api != null) num_platforms += 1;
        }
        if (num_platforms == 0) {
            log.err("No platform available", .{});
            return error.NoPlatformAvailable;
        }

        return .{ .platforms = PlatformsMap.initFill(null) };
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

    const prefered_targets = [_]Target{ .tpu, .neuron, .cuda, .rocm, .cpu };

    /// Automatically selects the best Platform loaded in the current Context.
    ///
    /// For example, if supported, this will select a platform corresponding to an accelerator (GPU, TPU, ...).
    pub fn autoPlatform(self: *Context, opts: Platform.CreateOptions) Platform {
        stdx.debug.assert(prefered_targets.len == apis.values.len, "New target need to be inserted inside `zml.Context.preferred_targets`", .{});

        return self.platformByPreferences(opts, &prefered_targets);
    }

    /// Given a list of preferred targets to select the best Platform
    ///
    /// For example, if supported, this will select a platform corresponding to an accelerator (GPU, TPU, ...).
    pub fn platformByPreferences(self: *Context, opts: Platform.CreateOptions, prefered: []const Target) Platform {
        // Try prefered targets.
        for (prefered) |target| {
            if (apis.get(target) == null) continue;
            return self.platform(target, opts) catch |err| {
                log.err("Failed to load platform .{s}: {}", .{ @tagName(target), err });
                continue;
            };
        }

        // Try unlisted targets
        var it = Context.apis.iterator();
        while (it.next()) |entry| {
            const target = entry.key;
            // CPU should only be use as fallback.
            if (target == .cpu) continue;
            if (entry.value.* == null) continue;
            if (std.mem.indexOfScalar(Target, prefered, target) != null) continue;
            return self.platform(target, opts) catch |err| {
                log.err("Failed to load platform .{s}: {}", .{ @tagName(target), err });
                continue;
            };
        }

        // Finally fallback to cpu.
        return self.platform(.cpu, opts) catch {
            log.err("No platform available", .{});
            @panic("No platform available !");
        };
    }

    pub fn platform(self: *Context, target: Target, opts: Platform.CreateOptions) !Platform {
        if (self.platforms.get(target)) |p| {
            return p;
        }
        const api = Context.apis.get(target);
        if (api == null) return error.PlatformNotCompiled;
        const p = try Platform.init(target, api.?, opts);
        if (p.getDevices().len == 0) {
            log.err("No device found for platform {} !", .{target});
            return error.NoDevicesFound;
        }

        try CustomCall.registerZmlCustomCalls(p);

        self.platforms.set(target, p);
        return p;
    }

    pub fn printAvailablePlatforms(self: Context, selected: Platform) void {
        // List available targets
        log.info("Available Platforms:", .{});
        const selected_prefix = "✅";
        const not_selected_prefix = "• ";
        const selected_postfix = "(AUTO-SELECTED)";
        const not_selected_postfix = "";

        for (zml_platform.available_targets) |target| {
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

    pub const HostCallback = fn (?*anyopaque, []const HostBuffer, []const HostBuffer) void;
};

const CustomCall = struct {
    pub fn registerZmlCustomCalls(platform: Platform) !void {
        const maybe_ffi = platform.pjrt_api.ffi();

        if (maybe_ffi) |ffi| {
            try ffi.register(platform.pjrt_api, "zmlHostBufferCallback", @tagName(platform.target), &hostBufferCallback, .{});
        } else {
            stdx.debug.panic("Registering custom calls failed", .{});
        }
    }

    fn hostBufferCallback(call_frame: *pjrt.ffi.CallFrame) callconv(.C) ?*pjrt.ffi.Error {
        if (call_frame.registeringHook()) return null;

        const callback_attr = call_frame.attrs.getByName(.scalar, "callback") orelse unreachable;
        std.debug.assert(callback_attr.dtype == .u64);
        const callback: *const Context.HostCallback = @ptrFromInt(callback_attr.get(usize));

        const user_ctx_ptr = call_frame.attrs.getByName(.scalar, "user_context") orelse unreachable;
        std.debug.assert(user_ctx_ptr.dtype == .u64);
        const user_ctx: ?*anyopaque = @ptrFromInt(user_ctx_ptr.get(usize));

        const input_buffers = stdx.stackSlice(8, HostBuffer, call_frame.args.len);
        for (input_buffers, 0..) |*b, i| {
            b.* = hostBufferFromPinnedBuffer(call_frame.args.buffers()[i]);
        }

        const output_buffers = stdx.stackSlice(8, HostBuffer, call_frame.results.len);
        for (output_buffers, 0..) |*b, i| {
            b.* = hostBufferFromPinnedBuffer(call_frame.results.buffers()[i]);
        }

        callback(user_ctx, input_buffers, output_buffers);
        return null;
    }
};

fn getShape(buffer_desc: *const pjrt.ffi.Buffer) Shape {
    // log.warn("received buffer {}", .{buffer_desc});
    const dt: DataType = switch (buffer_desc.dtype) {
        .invalid => @panic("invalid ffi"),
        .pred => .bool,
        .i8 => .i8,
        .i16 => .i16,
        .i32 => .i32,
        .i64 => .i64,
        .token, .f8e4m3, .f8e3m4 => @panic("Unsupported ffi type"),
        inline else => |t| @field(DataType, @tagName(t)),
    };
    return Shape.init(buffer_desc.dims(), dt);
}

/// Create a HostBuffer from a ffi description of a buffer.
/// Normally the ffi describe device buffer but we assume they are located in pinned memory,
/// and therefore the data pointer is readable both from host and from device.
fn hostBufferFromPinnedBuffer(buffer_desc: *const pjrt.ffi.Buffer) HostBuffer {
    const buffer_shape = getShape(buffer_desc);
    return HostBuffer.fromBytes(
        buffer_shape,
        buffer_desc.data[0..buffer_shape.byteSize()],
    );
}

pub const cuda = struct {
    pub var streamSynchronize: StreamSynchronize = @ptrFromInt(0xdeadc00da00);
    pub var cuLaunchHostFunc: CuLaunchHostFunc = @ptrFromInt(0xdeadc00da00);
    var _memcpyAsync: MemcpyAsync = @ptrFromInt(0xdeadc00da00);
    var _memcpyBlocking: MemcpyBlocking = @ptrFromInt(0xdeadc00da00);

    pub const MemcpyKind = enum(c_int) {
        host_to_host = 0,
        host_to_device = 1,
        device_to_host = 2,
        device_to_device = 3,
        inferred = 4,
    };

    const MemcpyAsync = *const fn (dst: *anyopaque, src: *const anyopaque, count: usize, kind: MemcpyKind, stream: ?*anyopaque) callconv(.C) c_int;
    const MemcpyBlocking = *const fn (dst: *anyopaque, src: *const anyopaque, count: usize, kind: MemcpyKind) callconv(.C) c_int;
    const StreamSynchronize = *const fn (stream: *anyopaque) callconv(.C) c_int;
    const CuLaunchHostFunc = *const fn (stream: *anyopaque, host_func: *const fn (user_data: *const anyopaque) callconv(.c) void, user_data: *const anyopaque) callconv(.c) c_int;

    pub fn init() void {
        var cudart = std.DynLib.open("libcudart.so.12") catch {
            log.err("cudart not found, callback will segfault", .{});
            return;
        };
        defer cudart.close();

        _memcpyAsync = cudart.lookup(MemcpyAsync, "cudaMemcpyAsync") orelse {
            @panic("cudaMemcpyAsync not found");
        };
        _memcpyBlocking = cudart.lookup(MemcpyBlocking, "cudaMemcpy") orelse {
            @panic("cudaMemcpy not found");
        };
        streamSynchronize = cudart.lookup(StreamSynchronize, "cudaStreamSynchronize") orelse {
            @panic("cudaStreamSynchronize not found");
        };
        cuLaunchHostFunc = cudart.lookup(CuLaunchHostFunc, "cudaLaunchHostFunc") orelse {
            @panic("cudaLaunchHostFunc not found");
        };
    }

    pub fn memcpyToHostBlocking(dst: []u8, src: *const anyopaque) void {
        const err = _memcpyBlocking(dst.ptr, src, dst.len, .device_to_host);
        check(err);
    }

    pub fn memcpyToDeviceBlocking(dst: *anyopaque, src: []const u8) void {
        const err = _memcpyBlocking(dst, src.ptr, src.len, .host_to_device);
        check(err);
    }

    pub fn memcpyToDeviceAsync(dst: *anyopaque, src: []const u8, stream: ?*anyopaque) void {
        const err = _memcpyAsync(dst, src.ptr, src.len, .host_to_device, stream);
        check(err);
    }

    pub fn memcpyToHostAsync(dst: []u8, src: *const anyopaque, stream: ?*anyopaque) void {
        const err = _memcpyAsync(dst.ptr, src, dst.len, .device_to_host, stream);
        check(err);
    }

    pub fn check(err: c_int) void {
        if (err == 0) return;
        stdx.debug.panic("CUDA error: {d}", .{err});
    }
};
