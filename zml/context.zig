const builtin = @import("builtin");
const std = @import("std");
const mlir = @import("mlir");
const asynk = @import("async");

const platform = @import("platform.zig");
const pjrtx = @import("pjrtx.zig");

const available_targets = @import("platform.zig").available_targets;
const Target = @import("platform.zig").Target;
const Platform = @import("platform.zig").Platform;

const log = std.log.scoped(.zml);

const PjrtApiMap = std.EnumArray(Target, ?*const pjrtx.Api);
const PlatformsMap = std.EnumArray(Target, ?Platform);

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
            inline for (platform.available_targets) |t| {
                if (canLoad(t)) {
                    if (pjrtx.Api.loadFrom(platformToLibrary(t))) |api| {
                        Context.apis.set(t, api);
                    } else |_| {}
                }
            }
        }
    }.call);

    var mlir_once = std.once(struct {
        fn call() void {
            mlir.registerPasses("Transforms");
        }
    }.call);

    platforms: PlatformsMap,

    /// Creates a ZML Context and returns it.
    pub fn init() !Context {
        Context.apis_once.call();
        Context.mlir_once.call();

        var platforms = PlatformsMap.initFill(null);
        var it = Context.apis.iterator();
        while (it.next()) |entry| {
            if (entry.value.*) |api| {
                const target = entry.key;
                const p = Platform.init(target, api) catch continue;
                if (p.getDevices().len == 0) {
                    log.err("No device found for platform {} !", .{target});
                    continue;
                }
                platforms.set(target, p);
            }
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

    fn canLoad(t: Target) bool {
        return switch (t) {
            .tpu => isRunningOnGCP() catch false,
            else => true,
        };
    }

    /// Check if running on Google Compute Engine, because TPUs will poll the
    /// metadata server, hanging the process. So only do it on GCP.
    /// Do it using the official method at:
    /// https://cloud.google.com/compute/docs/instances/detect-compute-engine?hl=en#use_operating_system_tools_to_detect_if_a_vm_is_running_in
    fn isRunningOnGCP() !bool {
        // TODO: abstract that in the client and fail init
        const GoogleComputeEngine = "Google Compute Engine";

        var f = try asynk.File.open("/sys/devices/virtual/dmi/id/product_name", .{ .mode = .read_only });
        defer f.close() catch {};

        var buf = [_]u8{0} ** GoogleComputeEngine.len;
        _ = try f.reader().readAll(&buf);

        return std.mem.eql(u8, &buf, GoogleComputeEngine);
    }

    pub fn pjrtApi(target: Target) *const pjrtx.Api {
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
        var platform_: Platform = undefined;
        var iterator = self.platforms.iterator();
        while (iterator.next()) |entry| {
            if (entry.value.*) |p| {
                platform_ = p;
            }
        }
        return platform_;
    }
};
