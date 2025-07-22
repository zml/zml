const std = @import("std");
const builtin = @import("builtin");

const c = @import("c");
const mlir = @import("mlir");
const runfiles = @import("runfiles");
const runtimes = @import("runtimes");
const stdx = @import("stdx");

const pjrt = @import("pjrtx.zig");

const zml = struct {
    pub const Platform = @import("platform.zig").Platform;
    pub const HostBuffer = @import("hostbuffer.zig").HostBuffer;
    pub const Shape = @import("shape.zig").Shape;
    pub const Target = @import("platform.zig").Target;
    pub const custom_call = @import("custom_call.zig");
    pub const platform = @import("platform.zig");
};

const PjrtApiMap = std.EnumArray(zml.Target, ?*const pjrt.Api);
const PlatformsMap = std.EnumArray(zml.Target, ?zml.Platform);
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

    fn platformToLibrary(comptime target: zml.Target) []const u8 {
        const ext = switch (builtin.os.tag) {
            .windows => ".dll",
            .macos, .ios, .watchos => ".dylib",
            else => ".so",
        };
        return switch (target) {
            inline else => "libpjrt_" ++ @tagName(target) ++ ext,
        };
    }

    pub fn pjrtApi(target: zml.Target) *const pjrt.Api {
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

    const prefered_targets = [_]zml.Target{ .tpu, .neuron, .cuda, .rocm, .cpu };

    /// Automatically selects the best Platform loaded in the current Context.
    ///
    /// For example, if supported, this will select a platform corresponding to an accelerator (GPU, TPU, ...).
    pub fn autoPlatform(self: *Context, opts: zml.Platform.CreateOptions) zml.Platform {
        stdx.debug.assert(prefered_targets.len == apis.values.len, "New target need to be inserted inside `zml.Context.preferred_targets`", .{});

        return self.platformByPreferences(opts, &prefered_targets);
    }

    /// Given a list of preferred targets to select the best Platform
    ///
    /// For example, if supported, this will select a platform corresponding to an accelerator (GPU, TPU, ...).
    pub fn platformByPreferences(self: *Context, opts: zml.Platform.CreateOptions, prefered: []const zml.Target) zml.Platform {
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
            if (std.mem.indexOfScalar(zml.Target, prefered, target) != null) continue;
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

    pub fn platform(self: *Context, target: zml.Target, opts: zml.Platform.CreateOptions) !Platform {
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

        self.platforms.set(target, p);
        try custom_call.registerInternalCustomCalls(p);

        return p;
    }

    pub fn printAvailablePlatforms(self: Context, selected: zml.Platform) void {
        // List available targets
        log.info("Available Platforms:", .{});
        const selected_prefix = "✅";
        const not_selected_prefix = "• ";
        const selected_postfix = "(AUTO-SELECTED)";
        const not_selected_postfix = "";

        for (zml.platform.available_targets) |target| {
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
};

const CustomCall = struct {
    pub fn registerZmlCustomCalls(platform: zml.Platform) !void {
        const ffi = platform.pjrt_api.ffi() orelse {
            log.warn("Registering custom calls failed: No FFI Extension found in {s} PJRT Plugin.", .{@tagName(platform.target)});
            return;
        };
        try ffi.register(platform.pjrt_api, "zmlHostBufferCallback", @tagName(platform.target), &hostBufferCallback, .{});
    }

    fn hostBufferCallback(call_frame: *pjrt.ffi.CallFrame) callconv(.C) ?*pjrt.ffi.Error {
        if (call_frame.registeringHook()) return null;

        const callback_attr = call_frame.attrs.getByName(.scalar, "callback") orelse unreachable;
        std.debug.assert(callback_attr.dtype == .u64);
        const callback: *const Context.HostCallback = @ptrFromInt(callback_attr.get(usize));

        const user_ctx_ptr = call_frame.attrs.getByName(.scalar, "user_context") orelse unreachable;
        std.debug.assert(user_ctx_ptr.dtype == .u64);
        const user_ctx: ?*anyopaque = @ptrFromInt(user_ctx_ptr.get(usize));

        const input_buffers = stdx.stackSlice(8, zml.HostBuffer, call_frame.args.len);
        for (input_buffers, 0..) |*b, i| {
            b.* = hostBufferFromPinnedBuffer(call_frame.args.buffers()[i]);
        }

        const output_buffers = stdx.stackSlice(8, zml.HostBuffer, call_frame.results.len);
        for (output_buffers, 0..) |*b, i| {
            b.* = hostBufferFromPinnedBuffer(call_frame.results.buffers()[i]);
        }

        callback(user_ctx, input_buffers, output_buffers);
        return null;
    }
};

fn getShape(buffer_desc: *const pjrt.ffi.Buffer) zml.Shape {
    // log.warn("received buffer {}", .{buffer_desc});
    const dt: zml.DataType = switch (buffer_desc.dtype) {
        .invalid => @panic("invalid ffi"),
        .pred => .bool,
        .i8 => .i8,
        .i16 => .i16,
        .i32 => .i32,
        .i64 => .i64,
        .token, .f8e4m3, .f8e3m4 => @panic("Unsupported ffi type"),
        inline else => |t| @field(zml.DataType, @tagName(t)),
    };
    return zml.Shape.init(buffer_desc.dims(), dt);
}

/// Create a HostBuffer from a ffi description of a buffer.
/// Normally the ffi describe device buffer but we assume they are located in pinned memory,
/// and therefore the data pointer is readable both from host and from device.
fn hostBufferFromPinnedBuffer(buffer_desc: *const pjrt.ffi.Buffer) zml.HostBuffer {
    const buffer_shape = getShape(buffer_desc);
    return zml.HostBuffer.fromBytes(
        buffer_shape,
        buffer_desc.data[0..buffer_shape.byteSize()],
    );
}
