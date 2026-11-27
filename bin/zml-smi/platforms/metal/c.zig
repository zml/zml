//! Direct `extern "c"` declarations for Apple's CoreFoundation, IOKit, and private IOReport APIs.
//! Linked via the `frameworks` cc_library target until rules_zig's `translate-c` supports
//! `-iframework <sysroot>/System/Library/Frameworks` or similar.

const std = @import("std");

// Mach / IOKit types and constants
pub const kern_return_t = std.c.kern_return_t;
pub const KERN_SUCCESS: kern_return_t = 0;

pub const io_object_t = std.c.mach_port_t;
pub const io_service_t = io_object_t;
pub const io_connect_t = io_object_t;
pub const io_iterator_t = io_object_t;
pub const io_registry_entry_t = io_object_t;

pub const kIOMainPortDefault: std.c.mach_port_t = 0;

// CoreFoundation types and constants
pub const CFString = opaque {};
pub const CFNumber = opaque {};
pub const CFArray = opaque {};
pub const CFDictionary = opaque {};
pub const CFData = opaque {};

pub const CFTypeRef = ?*const anyopaque;
pub const CFAllocatorRef = ?*const anyopaque;
pub const CFStringRef = ?*const CFString;
pub const CFNumberRef = ?*const CFNumber;
pub const CFArrayRef = ?*const CFArray;
pub const CFDictionaryRef = ?*const CFDictionary;
pub const CFMutableDictionaryRef = ?*CFDictionary;
pub const CFDataRef = ?*const CFData;

pub const CFIndex = isize;
pub const CFTypeID = usize;
pub const CFStringEncoding = u32;
pub const CFNumberType = CFIndex;
pub const Boolean = u8;

pub const kCFStringEncodingUTF8: CFStringEncoding = 0x08000100;
pub const kCFNumberSInt64Type: CFNumberType = 4;

// CoreFoundation
pub extern "c" fn CFRelease(cf: CFTypeRef) void;
pub extern "c" fn CFGetTypeID(cf: CFTypeRef) CFTypeID;

pub extern "c" fn CFStringGetTypeID() CFTypeID;
pub extern "c" fn CFStringCreateWithCString(alloc: CFAllocatorRef, c_str: [*:0]const u8, encoding: CFStringEncoding) CFStringRef;
pub extern "c" fn CFStringGetCString(the_string: CFStringRef, buffer: [*]u8, buffer_size: CFIndex, encoding: CFStringEncoding) Boolean;

pub extern "c" fn CFNumberGetTypeID() CFTypeID;
pub extern "c" fn CFNumberGetValue(number: CFNumberRef, the_type: CFNumberType, value_ptr: *anyopaque) Boolean;

pub extern "c" fn CFArrayGetTypeID() CFTypeID;
pub extern "c" fn CFArrayGetCount(the_array: CFArrayRef) CFIndex;
pub extern "c" fn CFArrayGetValueAtIndex(the_array: CFArrayRef, idx: CFIndex) CFTypeRef;

pub extern "c" fn CFDictionaryGetTypeID() CFTypeID;
pub extern "c" fn CFDictionaryCreateMutableCopy(allocator: CFAllocatorRef, capacity: CFIndex, the_dict: CFDictionaryRef) CFMutableDictionaryRef;
pub extern "c" fn CFDictionaryGetValue(the_dict: CFDictionaryRef, key: ?*const anyopaque) CFTypeRef;

pub extern "c" fn CFDataGetTypeID() CFTypeID;
pub extern "c" fn CFDataGetLength(the_data: CFDataRef) CFIndex;
pub extern "c" fn CFDataGetBytePtr(the_data: CFDataRef) ?[*]const u8;

// IOKit
pub extern "c" fn IOServiceMatching(name: [*:0]const u8) CFMutableDictionaryRef;
pub extern "c" fn IOServiceNameMatching(name: [*:0]const u8) CFMutableDictionaryRef;
pub extern "c" fn IOServiceGetMatchingService(main_port: std.c.mach_port_t, matching: CFDictionaryRef) io_service_t;
pub extern "c" fn IOServiceOpen(service: io_service_t, owning_task: std.c.mach_port_t, kind: u32, connect: *io_connect_t) kern_return_t;
pub extern "c" fn IOServiceClose(connect: io_connect_t) kern_return_t;
pub extern "c" fn IOConnectCallStructMethod(
    connection: std.c.mach_port_t,
    selector: u32,
    input_struct: *const anyopaque,
    input_struct_cnt: usize,
    output_struct: *anyopaque,
    output_struct_cnt: *usize,
) kern_return_t;
pub extern "c" fn IORegistryEntryGetChildIterator(entry: io_registry_entry_t, plane: [*:0]const u8, iterator: *io_iterator_t) kern_return_t;
pub extern "c" fn IORegistryEntryCreateCFProperty(entry: io_registry_entry_t, key: CFStringRef, allocator: CFAllocatorRef, options: u32) CFTypeRef;
pub extern "c" fn IOIteratorNext(iterator: io_iterator_t) io_object_t;
pub extern "c" fn IOObjectConformsTo(object: io_object_t, class_name: [*:0]const u8) c_int;
pub extern "c" fn IOObjectRelease(object: io_object_t) kern_return_t;

// Private IOReport (libIOReport.dylib)
pub const IOReportSubscription = opaque {};

pub extern "c" fn IOReportCopyChannelsInGroup(group: CFStringRef, subgroup: CFStringRef, a: u64, b: u64, c: u64) CFDictionaryRef;
pub extern "c" fn IOReportMergeChannels(into: CFMutableDictionaryRef, from: CFDictionaryRef, unused: CFTypeRef) void;
pub extern "c" fn IOReportCreateSubscription(a: ?*anyopaque, channels: CFMutableDictionaryRef, subscribed: *CFMutableDictionaryRef, d: u64, e: CFTypeRef) ?*IOReportSubscription;
pub extern "c" fn IOReportCreateSamples(subscription: *IOReportSubscription, channels: CFMutableDictionaryRef, unused: CFTypeRef) CFDictionaryRef;
pub extern "c" fn IOReportCreateSamplesDelta(previous: CFDictionaryRef, current: CFDictionaryRef, unused: CFTypeRef) CFDictionaryRef;
pub extern "c" fn IOReportChannelGetGroup(channel: CFDictionaryRef) CFStringRef;
pub extern "c" fn IOReportChannelGetChannelName(channel: CFDictionaryRef) CFStringRef;
pub extern "c" fn IOReportChannelGetUnitLabel(channel: CFDictionaryRef) CFStringRef;
pub extern "c" fn IOReportSimpleGetIntegerValue(channel: CFDictionaryRef, index: i32) i64;
pub extern "c" fn IOReportStateGetCount(channel: CFDictionaryRef) i32;
pub extern "c" fn IOReportStateGetNameForIndex(channel: CFDictionaryRef, index: i32) CFStringRef;
pub extern "c" fn IOReportStateGetResidency(channel: CFDictionaryRef, index: i32) i64;
