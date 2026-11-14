//! The IOKit functions Metal calls. The hermetic toolchain has no IOKit to link
//! against and no headers for it, so they are declared here and resolved by install name.
const std = @import("std");
const smi = @import("zml-smi");

/// An io_object_t: a service, an iterator or a connection. 0 is none.
pub const Object = std.c.mach_port_t;

/// kIOMainPortDefault.
pub const main_port: std.c.mach_port_t = 0;

pub const Api = struct {
    IOServiceMatching: *const @TypeOf(decls.IOServiceMatching),
    IOServiceNameMatching: *const @TypeOf(decls.IOServiceNameMatching),
    IOServiceGetMatchingService: *const @TypeOf(decls.IOServiceGetMatchingService),
    IOServiceOpen: Checked(decls.IOServiceOpen),
    IOServiceClose: *const @TypeOf(decls.IOServiceClose),
    IOConnectCallStructMethod: Checked(decls.IOConnectCallStructMethod),
    IORegistryEntryGetChildIterator: Checked(decls.IORegistryEntryGetChildIterator),
    IORegistryEntryCreateCFProperty: *const @TypeOf(decls.IORegistryEntryCreateCFProperty),
    IOIteratorNext: *const @TypeOf(decls.IOIteratorNext),
    IOObjectConformsTo: *const @TypeOf(decls.IOObjectConformsTo),
    IOObjectRelease: *const @TypeOf(decls.IOObjectRelease),
};

pub fn open() !Api {
    return smi.dynlib.open(Api, &.{"/System/Library/Frameworks/IOKit.framework/IOKit"});
}

fn Checked(comptime function: anytype) type {
    return smi.dynlib.Checked(function, check);
}

fn check(status: std.c.kern_return_t) error{IoKitError}!void {
    if (status != 0) return error.IoKitError;
}

/// Never called directly: `Api` takes their types.
const decls = struct {
    /// Returns a matching dictionary, which the lookup consumes.
    extern fn IOServiceMatching(name: [*:0]const u8) ?*anyopaque;
    extern fn IOServiceNameMatching(name: [*:0]const u8) ?*anyopaque;
    extern fn IOServiceGetMatchingService(port: std.c.mach_port_t, matching: ?*anyopaque) Object;
    extern fn IOServiceOpen(service: Object, task: std.c.mach_port_t, kind: u32, connection: *Object) std.c.kern_return_t;
    extern fn IOServiceClose(connection: Object) std.c.kern_return_t;
    extern fn IOConnectCallStructMethod(connection: Object, selector: u32, input: *const anyopaque, input_size: usize, output: *anyopaque, output_size: *usize) std.c.kern_return_t;
    extern fn IORegistryEntryGetChildIterator(entry: Object, plane: [*:0]const u8, iterator: *Object) std.c.kern_return_t;
    /// Returns a CoreFoundation object the caller releases.
    extern fn IORegistryEntryCreateCFProperty(entry: Object, key: *const anyopaque, allocator: ?*const anyopaque, options: u32) ?*const anyopaque;
    extern fn IOIteratorNext(iterator: Object) Object;
    extern fn IOObjectConformsTo(object: Object, class: [*:0]const u8) c_int;
    extern fn IOObjectRelease(object: Object) std.c.kern_return_t;
};
