//! The IOKit functions Metal calls. The hermetic toolchain has no IOKit to link
//! against and no headers for it, so they are declared here and resolved by install name.
const std = @import("std");
const native = @import("zml-smi-2/native");

pub fn CFunction(comptime parameters: []const type, comptime Result: type) type {
    return *const @Fn(parameters, &@splat(.{}), Result, .{ .@"callconv" = .c });
}

fn Checked(comptime parameters: []const type) type {
    return native.dynlib.Checked(check, std.meta.Child(CFunction(parameters, Status)));
}

/// An io_object_t: a service, an iterator or a connection. 0 is none.
pub const Object = std.c.mach_port_t;
const Status = std.c.kern_return_t;
const AnyRef = *const anyopaque;

/// kIOMainPortDefault.
pub const main_port: std.c.mach_port_t = 0;

pub const Api = struct {
    /// Returns a matching dictionary, which the lookup consumes.
    IOServiceMatching: Matching,
    IOServiceNameMatching: Matching,
    IOServiceGetMatchingService: CFunction(&.{ std.c.mach_port_t, ?*anyopaque }, Object),
    IOServiceOpen: Checked(&.{ Object, std.c.mach_port_t, u32, *Object }),
    IOServiceClose: CFunction(&.{Object}, Status),
    IOConnectCallStructMethod: Checked(&.{ Object, u32, AnyRef, usize, *anyopaque, *usize }),
    IORegistryEntryGetChildIterator: Checked(&.{ Object, [*:0]const u8, *Object }),
    /// Returns a CoreFoundation object the caller releases.
    IORegistryEntryCreateCFProperty: CFunction(&.{ Object, AnyRef, ?AnyRef, u32 }, ?AnyRef),
    IOIteratorNext: CFunction(&.{Object}, Object),
    IOObjectConformsTo: CFunction(&.{ Object, [*:0]const u8 }, c_int),
    IOObjectRelease: CFunction(&.{Object}, Status),

    const Matching = CFunction(&.{[*:0]const u8}, ?*anyopaque);
};

pub fn open() !Api {
    return native.dynlib.open(Api, &.{"/System/Library/Frameworks/IOKit.framework/IOKit"});
}

fn check(status: Status) error{IoKitError}!void {
    if (status != 0) return error.IoKitError;
}
