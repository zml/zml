const std = @import("std");
const builtin = @import("builtin");

const c = @import("c");
const ffi = @import("xlaffi");
const frame_info = ffi.frame_info;
const mlir = @import("mlir");
const runfiles = @import("runfiles");
const runtimes = @import("runtimes");
const stdx = @import("stdx");

const available_targets = @import("platform.zig").available_targets;
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

    pub const HostCallbackCtx = struct {
        host: HostBuffer,
        platform: Platform,
        mutex: std.Thread.Mutex = std.Thread.Mutex{},
    };
    pub const HostCallback = fn (HostBuffer) void;
    pub const DeviceCallback = fn (Platform, ?*anyopaque, []const HostBuffer, []Buffer) void;
};

const CustomCall = struct {
    pub fn registerZmlCustomCalls(platform: Platform) !void {
        const registry = platform.pjrt_api.customCallRegistry();

        if (registry) |reg| {
            try reg.registerFfi(platform.pjrt_api, "zmlDeviceBufferCallback", @tagName(platform.target), &deviceBufferCallback);
            try reg.registerFfi(platform.pjrt_api, "zmlHostBufferCallback", @tagName(platform.target), &hostBufferCallback);
        } else {
            stdx.debug.panic("Registering custom calls failed", .{});
        }
    }

    fn getContext(attrs: ffi.Attrs) struct { *const Context.HostCallback, *Context.HostCallbackCtx } {
        const context_scalar = attrs.getAttrByNameAs(ffi.Scalar, "context") orelse unreachable;
        std.debug.assert(context_scalar.dtype == .s64);
        const ctx: *Context.HostCallbackCtx = @ptrFromInt(@as(usize, @bitCast(@as(*i64, @ptrCast(@alignCast(context_scalar.value))).*)));

        const callback_scalar = attrs.getAttrByNameAs(ffi.Scalar, "callback") orelse unreachable;
        std.debug.assert(callback_scalar.dtype == .s64);
        const callback: *const Context.HostCallback = @ptrFromInt(@as(usize, @bitCast(@as(*i64, @ptrCast(@alignCast(callback_scalar.value))).*)));

        return .{ callback, ctx };
    }

    fn hostBufferCallback(call_frame: *ffi.CallFrame) callconv(.C) ?*ffi.Error {
        if (call_frame.extension_start != null and call_frame.extension_start.?.type == .metadata) {
            const metadata_extension: *ffi.MetadataExtension = @fieldParentPtr("extension_base", call_frame.extension_start.?);
            metadata_extension.metadata.?.api_version.major_version = ffi.ApiVersion.major;
            metadata_extension.metadata.?.api_version.minor_version = ffi.ApiVersion.minor;
            return null;
        }

        // Print frame info to stderr
        frame_info.printCallFrameInfo(call_frame, std.io.getStdErr().writer()) catch {};

        // If you have a buffer argument:
        const buffer = call_frame.args.getArgAs(ffi.Buffer, 0);
        frame_info.getBufferInfo(buffer, std.io.getStdErr().writer()) catch {};

        const callback, const ctx = getContext(call_frame.attrs);
        // Add synchronization because this is called from the device driver.
        ctx.mutex.lock();
        defer ctx.mutex.unlock();
        const ffi_buffer = call_frame.args.getArgAs(ffi.Buffer, 0);

        const MAX_RANK: u8 = 8;

        const minor_to_major: [MAX_RANK]i64 = comptime blk: {
            var res: [MAX_RANK]i64 = undefined;
            for (0..MAX_RANK) |i| {
                res[i] = @intCast(MAX_RANK - i - 1);
            }
            break :blk res;
        };

        const pjrt_buffer = ctx.platform.pjrt_client.createViewOfDeviceBuffer2(ctx.platform.pjrt_api, .{
            .device_buffer_ptr = ffi_buffer.data,
            .element_type = .F32,
            .dims = ffi_buffer.dims(),
            .device = ctx.platform.getDevices()[0],
            .layout = .{
                .tiled = .{
                    .minor_to_major = minor_to_major[@as(usize, MAX_RANK - @as(usize, @intCast(ffi_buffer.rank)))..],
                    .tile_dims = &.{},
                    .tile_dims_sizes = &.{},
                },
            },
        }) catch unreachable;

        const event = pjrt_buffer.toHostBuffer(ctx.platform.pjrt_api, @constCast(ctx.host.data)) catch |err| {
            log.err("Failed to copy buffer to host: {}", .{err});
            return null;
        };
        _ = event; // autofix

        std.debug.print("data: {any}\n", .{ctx.host.data[0..512]});

        callback(ctx.host);
        return null;
    }

    fn deviceBufferCallback(call_frame: *ffi.CallFrame) callconv(.C) ?*ffi.Error {
        if (call_frame.extension_start != null and call_frame.extension_start.?.type == .metadata) {
            const metadata_extension: *ffi.MetadataExtension = @fieldParentPtr("extension_base", call_frame.extension_start.?);
            metadata_extension.metadata.?.api_version.major_version = ffi.ApiVersion.major;
            metadata_extension.metadata.?.api_version.minor_version = ffi.ApiVersion.minor;
            return null;
        }

        // frame_info.printCallFrameInfo(call_frame, std.io.getStdErr().writer()) catch {};

        const callback_scalar = call_frame.attrs.getAttrByNameAs(ffi.Scalar, "callback") orelse unreachable;
        std.debug.assert(callback_scalar.dtype == .u64);
        const callback: *const Context.DeviceCallback = @ptrFromInt(callback_scalar.get(usize));

        const platform_ptr = call_frame.attrs.getAttrByNameAs(ffi.Scalar, "platform_ptr") orelse unreachable;
        std.debug.assert(platform_ptr.dtype == .u64);
        const platform: *const Platform = @ptrFromInt(platform_ptr.get(usize));

        const user_ctx_ptr = call_frame.attrs.getAttrByNameAs(ffi.Scalar, "user_context") orelse unreachable;
        std.debug.assert(user_ctx_ptr.dtype == .u64);
        const user_ctx: ?*anyopaque = @ptrFromInt(user_ctx_ptr.get(usize));

        const n_args: usize = @intCast(call_frame.args.size);
        const input_buffers = stdx.stackSlice(8, HostBuffer, n_args);
        for (input_buffers, 0..) |*b, i| {
            const buffer_desc = call_frame.args.getArgAs(ffi.Buffer, i);
            // log.warn("received buffer {}", .{buffer_desc});
            const dt: DataType = switch (buffer_desc.dtype) {
                .invalid => @panic("invalid ffi"),
                .pred => .bool,
                .s8 => .i8,
                .s16 => .i16,
                .s32 => .i32,
                .s64 => .i64,
                .token, .f8e4m3, .f8e3m4 => @panic("Unsupported ffi type"),
                inline else => |t| @field(DataType, @tagName(t)),
            };
            const buffer_shape = Shape.init(buffer_desc.dims(), dt);

            b.* = HostBuffer.fromBytes(buffer_shape, buffer_desc.data[0..buffer_shape.byteSize()]);
        }

        const n_ret = call_frame.rets.size;
        const output_buffers = stdx.stackSlice(8, Buffer, n_ret);
        callback(platform.*, user_ctx, input_buffers, output_buffers);
        for (output_buffers, 0..) |res, i| {
            const result_ptr = call_frame.rets.getRetAs(ffi.Buffer, i);
            // log.warn("writing {} to {}", .{ res, result_ptr.* });
            result_ptr.data = @ptrCast(res._shards.get(0).getOpaqueDeviceMemoryDataPointer(res._api) catch @panic("pjrt error"));
        }
        return null;
    }
};
