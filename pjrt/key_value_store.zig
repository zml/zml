const std = @import("std");

const c = @import("c");

pub const Error = error{
    DeadlineExceeded,
    Internal,
    NotFound,
    OutOfMemory,
    Unavailable,
};

/// Thread-safe key-value store used while creating distributed PJRT clients.
///
/// The store and `user_arg` must outlive the PJRT client. Values returned by
/// `get` and `try_get` must use the provided allocator. PJRT takes ownership
/// of successful results.
pub const KeyValueStore = struct {
    user_arg: *anyopaque,
    get: *const fn (
        user_arg: *anyopaque,
        allocator: std.mem.Allocator,
        key: []const u8,
        timeout_ms: c_int,
    ) Error![]u8,
    try_get: *const fn (
        user_arg: *anyopaque,
        allocator: std.mem.Allocator,
        key: []const u8,
    ) Error!?[]u8,
    put: *const fn (
        user_arg: *anyopaque,
        key: []const u8,
        value: []const u8,
    ) Error!void,
};

fn callbackError(
    callback_error: [*c]c.PJRT_CallbackError,
    err: Error,
) ?*c.PJRT_Error {
    const message = @errorName(err);
    return callback_error.*.?(
        switch (err) {
            error.DeadlineExceeded => c.PJRT_Error_Code_DEADLINE_EXCEEDED,
            error.NotFound => c.PJRT_Error_Code_NOT_FOUND,
            error.OutOfMemory => c.PJRT_Error_Code_RESOURCE_EXHAUSTED,
            error.Unavailable => c.PJRT_Error_Code_UNAVAILABLE,
            error.Internal => c.PJRT_Error_Code_INTERNAL,
        },
        message.ptr,
        message.len,
    );
}

fn free(value: [*c]u8) callconv(.c) void {
    if (value != null) std.c.free(@ptrCast(value));
}

fn setValue(args: anytype, value: []u8) Error!void {
    // PJRT calls the required deleter even for empty successful values.
    args.*.value = if (value.len == 0)
        @ptrCast((try std.heap.c_allocator.alloc(u8, 1)).ptr)
    else
        @ptrCast(value.ptr);
    args.*.value_size = value.len;
    args.*.value_deleter_callback = free;
}

pub fn getCallback(
    args: [*c]c.PJRT_KeyValueGetCallback_Args,
) callconv(.c) ?*c.PJRT_Error {
    const store: *const KeyValueStore = @ptrCast(
        @alignCast(args.*.user_arg.?),
    );
    const key = args.*.key[0..args.*.key_size];
    const value = store.get(
        store.user_arg,
        std.heap.c_allocator,
        key,
        args.*.timeout_in_ms,
    ) catch |err| return callbackError(args.*.callback_error, err);

    setValue(args, value) catch |err| {
        return callbackError(args.*.callback_error, err);
    };
    return null;
}

pub fn tryGetCallback(
    args: [*c]c.PJRT_KeyValueTryGetCallback_Args,
) callconv(.c) ?*c.PJRT_Error {
    const store: *const KeyValueStore = @ptrCast(
        @alignCast(args.*.user_arg.?),
    );
    const key = args.*.key[0..args.*.key_size];
    const value = store.try_get(
        store.user_arg,
        std.heap.c_allocator,
        key,
    ) catch |err| return callbackError(args.*.callback_error, err);

    const result = value orelse return callbackError(
        args.*.callback_error,
        error.NotFound,
    );
    setValue(args, result) catch |err| {
        return callbackError(args.*.callback_error, err);
    };
    return null;
}

pub fn putCallback(
    args: [*c]c.PJRT_KeyValuePutCallback_Args,
) callconv(.c) ?*c.PJRT_Error {
    const store: *const KeyValueStore = @ptrCast(
        @alignCast(args.*.user_arg.?),
    );
    const key = args.*.key[0..args.*.key_size];
    const value = args.*.value[0..args.*.value_size];
    store.put(store.user_arg, key, value) catch |err| {
        return callbackError(args.*.callback_error, err);
    };
    return null;
}

test "empty callback values have an owned non-null pointer" {
    const Args = struct {
        value: [*c]u8 = undefined,
        value_size: usize = undefined,
        value_deleter_callback: ?*const fn (
            value: [*c]u8,
        ) callconv(.c) void = undefined,
    };

    const empty = try std.heap.c_allocator.alloc(u8, 0);
    defer std.heap.c_allocator.free(empty);
    var args: Args = .{};
    try setValue(&args, empty);
    try std.testing.expect(args.value != null);
    try std.testing.expectEqual(0, args.value_size);
    args.value_deleter_callback.?(args.value);

    const value = try std.heap.c_allocator.dupe(u8, "value");
    try setValue(&args, value);
    try std.testing.expect(args.value != null);
    try std.testing.expectEqual(value.len, args.value_size);
    args.value_deleter_callback.?(args.value);
}
