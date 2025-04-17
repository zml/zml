const std = @import("std");
const builtin = @import("builtin");

const c = @import("c");
const mlir = @import("mlir");
const runfiles = @import("runfiles");
const runtimes = @import("runtimes");
const stdx = @import("stdx");

const Buffer = @import("buffer.zig").Buffer;
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

var _active_platform: Platform = undefined;

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
                if (t == .cuda) cuda.init();
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

        const p = self.platformByPreferences(opts, &prefered_targets);
        // TODO properly propagate the platform to the callbacks
        _active_platform = p;
        return p;
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
        const registry = platform.pjrt_api.customCallRegistry();

        if (registry) |reg| {
            try reg.registerFfi(platform.pjrt_api, "zmlHostBufferCallback", @tagName(platform.target), &hostBufferCallback);
        } else {
            stdx.debug.panic("Registering custom calls failed", .{});
        }
    }

    fn hostBufferCallback(call_frame: *pjrt.ffi.CallFrame) callconv(.C) ?*pjrt.ffi.Error {
        if (call_frame.registeringHook()) return null;

        // TODO correctly pass the callback w/o embedding ptr in MLIR
        const err = "malformed zmlHostBufferCallback custom call";
        const callback_attr = call_frame.attrs.getByName(.scalar, "callback") orelse @panic(err);
        std.debug.assert(callback_attr.dtype == .u64);
        const callback: *const Context.HostCallback = @ptrFromInt(callback_attr.get(usize));

        const user_ctx_ptr = call_frame.attrs.getByName(.scalar, "user_context") orelse @panic(err);
        std.debug.assert(user_ctx_ptr.dtype == .u64);
        const user_ctx: ?*anyopaque = @ptrFromInt(user_ctx_ptr.get(usize));

        const tags_arr = call_frame.attrs.getByName(.array, "__tags") orelse @panic(err);
        std.debug.assert(tags_arr.dtype == .i64);
        const tags = tags_arr.slice([*:0]const u8);

        const input_buffers = stdx.stackSlice(8, HostBuffer, call_frame.args.len);
        const output_buffers = stdx.stackSlice(8, HostBuffer, call_frame.results.len);

        // TODO correctly pass the platform w/o global
        const platform = _active_platform;
        if (platform.target == .cpu) {
            for (input_buffers, 0..) |*b, i| {
                b.* = hostViewOfDeviceBuffer(call_frame.args.get(i));
            }
            for (output_buffers, 0..) |*b, i| {
                b.* = hostViewOfDeviceBuffer(call_frame.results.get(i));
            }
            applyTags(tags, input_buffers, output_buffers);

            callback(user_ctx, input_buffers, output_buffers);
            return null;
        }

        if (platform.target != .cuda) @panic("hostBufferCallback are only supported on cpu and cuda for now");
        const stream = call_frame.*.api.stream(call_frame.ctx);
        for (input_buffers, 0..) |*b, i| {
            const d = call_frame.args.get(i);
            b.* = HostBuffer.empty(std.heap.smp_allocator, getShape(d)) catch @panic("OOM");
            const memcpy_err = cuda.memcpyAsync(@constCast(b.data.ptr), d.data, b.shape().byteSize(), .device_to_host, stream);
            std.debug.assert(memcpy_err == 0);
        }
        for (output_buffers, 0..) |*b, i| {
            const d = call_frame.results.get(i);
            b.* = HostBuffer.empty(std.heap.smp_allocator, getShape(d)) catch @panic("OOM");
        }
        applyTags(tags, input_buffers, output_buffers);

        const d2h_err = cuda.streamSynchronize(stream);
        std.debug.assert(d2h_err == 0);
        callback(user_ctx, input_buffers, output_buffers);

        for (output_buffers, 0..) |*b, i| {
            const d = call_frame.args.get(i);
            b.deinit(std.heap.smp_allocator);
            const memcpy_err = cuda.memcpyAsync(d.data, b.data.ptr, b.shape().byteSize(), .host_to_device, stream);
            std.debug.assert(memcpy_err == 0);
        }
        for (input_buffers) |*b| {
            b.deinit(std.heap.smp_allocator);
        }
        const h2d_err = cuda.streamSynchronize(stream);
        std.debug.assert(h2d_err == 0);

        return null;
    }
};

const cuda = struct {
    var memcpyAsync: MemcpyAsync = @ptrFromInt(0xdeadc00da00);
    var streamSynchronize: StreamSynchronize = @ptrFromInt(0xdeadc00da00);

    pub const MemcpyKind = enum(c_int) {
        host_to_host = 0,
        host_to_device = 1,
        device_to_host = 2,
        device_to_device = 3,
    };

    const MemcpyAsync = *const fn (dst: *anyopaque, src: *const anyopaque, count: usize, kind: MemcpyKind, stream: ?*pjrt.ffi.Stream) callconv(.C) c_int;
    const StreamSynchronize = *const fn (stream: ?*pjrt.ffi.Stream) callconv(.C) c_int;

    pub fn init() void {
        var cudart = std.DynLib.open("libcudart.so.12") catch {
            log.err("cudart not found, callback will segfault", .{});
            return;
        };
        defer cudart.close();

        memcpyAsync = cudart.lookup(MemcpyAsync, "cudaMemcpyAsync") orelse {
            @panic("cudaMemcpyAsync not found");
        };
        streamSynchronize = cudart.lookup(StreamSynchronize, "cudaStreamSynchronize") orelse {
            @panic("cudaStreamSynchronize not found");
        };
    }
};

fn getShape(buffer_desc: *const pjrt.ffi.Buffer) Shape {
    // log.warn("received buffer {}", .{buffer_desc});
    const dt: DataType = switch (buffer_desc.dtype) {
        .invalid => @panic("invalid ffi"),
        .pred => .bool,
        .token, .f8e4m3, .f8e3m4 => @panic("Unsupported ffi type"),
        inline else => |t| @field(DataType, @tagName(t)),
    };
    return Shape.init(buffer_desc.dims(), dt);
}

/// Create a Device from a ffi description of a buffer.
/// Normally the ffi describe device buffer but when platform == cpu it's the same addresspace
fn hostViewOfDeviceBuffer(buffer_desc: *const pjrt.ffi.Buffer) HostBuffer {
    const buffer_shape = getShape(buffer_desc);
    return HostBuffer.fromBytes(
        buffer_shape,
        buffer_desc.data[0..buffer_shape.byteSize()],
    );
}

fn deviceBuffer(platform: Platform, buffer_desc: *const pjrt.ffi.Buffer, stream: *const pjrt.ffi.Stream) Buffer {
    const buffer_shape = getShape(buffer_desc);
    return Buffer.asViewOfDeviceBuffer(platform, buffer_shape, stream, buffer_desc.data);
}

fn applyTags(flat_tags: []const [*:0]const u8, in: []HostBuffer, out: []HostBuffer) void {
    var tags = flat_tags;
    for (in) |*b| {
        for (b._shape._tags.slice()) |*t| {
            t.* = tags[0];
            tags = tags[1..];
        }
    }

    for (out) |*b| {
        for (b._shape._tags.slice()) |*t| {
            t.* = tags[0];
            tags = tags[1..];
        }
    }

    std.debug.assert(tags.len == 0);
}
