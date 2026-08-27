const std = @import("std");

const pjrt = @import("pjrt");

const log = std.log.scoped(.zml_distributed);

const magic = "ZMKV";
const protocol_version: u16 = 1;

pub const Error = error{
    Canceled,
    DeadlineExceeded,
    Internal,
    InvalidConfiguration,
    InvalidLifecycle,
    InvalidRequest,
    OutOfMemory,
    ResourceExhausted,
    Unavailable,
    UnsupportedVersion,
};

pub const Options = struct {
    startup_timeout: std.Io.Duration,
    operation_timeout: std.Io.Duration,
    retry_delay: std.Io.Duration,
    max_key_bytes: usize,
    max_value_bytes: usize,
    max_connections: usize,
};

const Operation = enum(u8) {
    get = 1,
    put = 2,
};

const Status = enum(u8) {
    ok = 0,
    not_found = 1,
    invalid_request = 2,
    resource_exhausted = 3,
    unsupported_version = 4,
    internal = 5,
};

const Selection = union(enum) {
    request: Error!?[]u8,
    timeout: std.Io.Cancelable!void,
};

pub const Client = struct {
    io: std.Io,
    address: std.Io.net.IpAddress,
    namespace: []const u8,
    namespace_hash: u64,
    options: Options,

    pub fn init(
        io: std.Io,
        address: std.Io.net.IpAddress,
        namespace: []const u8,
        options: Options,
    ) Client {
        return .{
            .io = io,
            .address = address,
            .namespace = namespace,
            .namespace_hash = std.hash.Wyhash.hash(0, namespace),
            .options = options,
        };
    }

    pub fn keyValueStore(self: *Client) pjrt.KeyValueStore {
        return .{
            .user_arg = self,
            .get = getCallback,
            .try_get = tryGetCallback,
            .put = putCallback,
        };
    }

    pub fn get(
        self: *const Client,
        allocator: std.mem.Allocator,
        key: []const u8,
        timeout: std.Io.Duration,
    ) Error![]u8 {
        return self.getUntil(
            allocator,
            key,
            deadlineFromNow(self.io, timeout),
        );
    }

    pub fn getUntil(
        self: *const Client,
        allocator: std.mem.Allocator,
        key: []const u8,
        deadline: std.Io.Timestamp,
    ) Error![]u8 {
        const started = std.Io.Clock.awake.now(self.io);
        const result = self.getUntilUnlogged(
            allocator,
            key,
            deadline,
        ) catch |err| {
            self.logOperation(
                .get,
                key.len,
                0,
                started,
                @errorName(err),
            );
            return err;
        };
        self.logOperation(.get, key.len, 0, started, "ok");
        return result;
    }

    fn getUntilUnlogged(
        self: *const Client,
        allocator: std.mem.Allocator,
        key: []const u8,
        deadline: std.Io.Timestamp,
    ) Error![]u8 {
        while (true) {
            const result = self.request(
                .get,
                key,
                "",
                allocator,
                deadline,
            ) catch |err| switch (err) {
                error.Unavailable => {
                    try self.waitToRetry(deadline);
                    continue;
                },
                else => return err,
            };
            if (result) |value| return value;
            try self.waitToRetry(deadline);
        }
    }

    pub fn tryGet(
        self: *const Client,
        allocator: std.mem.Allocator,
        key: []const u8,
    ) Error!?[]u8 {
        const started = std.Io.Clock.awake.now(self.io);
        const result = self.request(
            .get,
            key,
            "",
            allocator,
            deadlineFromNow(self.io, self.options.operation_timeout),
        ) catch |err| {
            self.logOperation(
                .get,
                key.len,
                0,
                started,
                @errorName(err),
            );
            return err;
        };
        self.logOperation(
            .get,
            key.len,
            0,
            started,
            if (result == null) "not_found" else "ok",
        );
        return result;
    }

    pub fn put(
        self: *const Client,
        key: []const u8,
        value: []const u8,
    ) Error!void {
        return self.putUntil(
            key,
            value,
            deadlineFromNow(self.io, self.options.operation_timeout),
        );
    }

    pub fn putUntil(
        self: *const Client,
        key: []const u8,
        value: []const u8,
        deadline: std.Io.Timestamp,
    ) Error!void {
        const started = std.Io.Clock.awake.now(self.io);
        self.putUntilUnlogged(key, value, deadline) catch |err| {
            self.logOperation(
                .put,
                key.len,
                value.len,
                started,
                @errorName(err),
            );
            return err;
        };
        self.logOperation(.put, key.len, value.len, started, "ok");
    }

    fn putUntilUnlogged(
        self: *const Client,
        key: []const u8,
        value: []const u8,
        deadline: std.Io.Timestamp,
    ) Error!void {
        while (true) {
            _ = self.request(
                .put,
                key,
                value,
                null,
                deadline,
            ) catch |err| switch (err) {
                error.Unavailable => {
                    try self.waitToRetry(deadline);
                    continue;
                },
                else => return err,
            };
            return;
        }
    }

    fn getCallback(
        user_arg: *anyopaque,
        allocator: std.mem.Allocator,
        key: []const u8,
        timeout_ms: c_int,
    ) pjrt.KeyValueStoreError![]u8 {
        const self: *const Client = @ptrCast(@alignCast(user_arg));
        const timeout = if (timeout_ms <= 0)
            self.options.startup_timeout
        else
            std.Io.Duration.fromMilliseconds(timeout_ms);
        return self.get(allocator, key, timeout) catch |err| {
            return pjrtError(err);
        };
    }

    fn tryGetCallback(
        user_arg: *anyopaque,
        allocator: std.mem.Allocator,
        key: []const u8,
    ) pjrt.KeyValueStoreError!?[]u8 {
        const self: *const Client = @ptrCast(@alignCast(user_arg));
        return self.tryGet(allocator, key) catch |err| {
            return pjrtError(err);
        };
    }

    fn putCallback(
        user_arg: *anyopaque,
        key: []const u8,
        value: []const u8,
    ) pjrt.KeyValueStoreError!void {
        const self: *const Client = @ptrCast(@alignCast(user_arg));
        return self.putUntil(
            key,
            value,
            deadlineFromNow(self.io, self.options.startup_timeout),
        ) catch |err| {
            return pjrtError(err);
        };
    }

    fn request(
        self: *const Client,
        operation: Operation,
        key: []const u8,
        value: []const u8,
        allocator: ?std.mem.Allocator,
        deadline: std.Io.Timestamp,
    ) Error!?[]u8 {
        try self.validateRequest(key, value);
        return self.requestUntil(
            operation,
            key,
            value,
            allocator,
            deadline,
        );
    }

    fn logOperation(
        self: *const Client,
        operation: Operation,
        key_bytes: usize,
        value_bytes: usize,
        started: std.Io.Timestamp,
        status: []const u8,
    ) void {
        log.debug(
            "operation={s} endpoint={f} namespace={x}" ++
                " key_bytes={d} value_bytes={d}" ++
                " duration_ms={d} status={s}",
            .{
                @tagName(operation),
                self.address,
                self.namespace_hash,
                key_bytes,
                value_bytes,
                started.untilNow(self.io, .awake).toMilliseconds(),
                status,
            },
        );
    }

    fn requestUntil(
        self: *const Client,
        operation: Operation,
        key: []const u8,
        value: []const u8,
        allocator: ?std.mem.Allocator,
        deadline: std.Io.Timestamp,
    ) Error!?[]u8 {
        if (deadlineReached(self.io, deadline)) {
            return error.DeadlineExceeded;
        }

        var result_buffer: [2]Selection = undefined;
        var select: std.Io.Select(Selection) = .init(
            self.io,
            &result_buffer,
        );
        select.async(
            .request,
            requestOnce,
            .{ self, operation, key, value, allocator },
        );
        select.async(.timeout, waitUntil, .{ self.io, deadline });

        const selected = select.await() catch {
            cancelRequest(&select, allocator);
            return error.Canceled;
        };
        switch (selected) {
            .request => |result| {
                select.cancelDiscard();
                return result;
            },
            .timeout => |result| {
                result catch {
                    cancelRequest(&select, allocator);
                    return error.Canceled;
                };
                cancelRequest(&select, allocator);
                return error.DeadlineExceeded;
            },
        }
    }

    fn requestOnce(
        self: *const Client,
        operation: Operation,
        key: []const u8,
        value: []const u8,
        allocator: ?std.mem.Allocator,
    ) Error!?[]u8 {
        const stream = self.address.connect(self.io, .{
            .mode = .stream,
        }) catch |err| return transportError(err);
        defer stream.close(self.io);

        var write_buffer: [1024]u8 = undefined;
        var stream_writer = stream.writer(self.io, &write_buffer);
        const writer = &stream_writer.interface;
        writer.writeAll(magic) catch {
            return writerError(stream_writer.err);
        };
        writer.writeInt(u16, protocol_version, .big) catch {
            return writerError(stream_writer.err);
        };
        writer.writeInt(u8, @intFromEnum(operation), .big) catch {
            return writerError(stream_writer.err);
        };
        writer.writeInt(u8, 0, .big) catch {
            return writerError(stream_writer.err);
        };
        writer.writeInt(u16, @intCast(self.namespace.len), .big) catch {
            return writerError(stream_writer.err);
        };
        writer.writeInt(u32, @intCast(key.len), .big) catch {
            return writerError(stream_writer.err);
        };
        writer.writeInt(u32, @intCast(value.len), .big) catch {
            return writerError(stream_writer.err);
        };
        writer.writeAll(self.namespace) catch {
            return writerError(stream_writer.err);
        };
        writer.writeAll(key) catch return writerError(stream_writer.err);
        writer.writeAll(value) catch return writerError(stream_writer.err);
        writer.flush() catch return writerError(stream_writer.err);

        var read_buffer: [1024]u8 = undefined;
        var stream_reader = stream.reader(self.io, &read_buffer);
        const reader = &stream_reader.interface;
        var response_magic: [magic.len]u8 = undefined;
        reader.readSliceAll(&response_magic) catch {
            return readerError(stream_reader.err);
        };
        if (!std.mem.eql(u8, magic, &response_magic)) {
            return error.InvalidRequest;
        }
        const version = reader.takeInt(u16, .big) catch {
            return readerError(stream_reader.err);
        };
        if (version != protocol_version) return error.UnsupportedVersion;
        const status_int = reader.takeInt(u8, .big) catch {
            return readerError(stream_reader.err);
        };
        const reserved = reader.takeInt(u8, .big) catch {
            return readerError(stream_reader.err);
        };
        if (reserved != 0) return error.InvalidRequest;
        const result_size = reader.takeInt(u32, .big) catch {
            return readerError(stream_reader.err);
        };
        const status = std.enums.fromInt(Status, status_int) orelse {
            return error.InvalidRequest;
        };
        if (result_size > self.options.max_value_bytes) {
            return error.ResourceExhausted;
        }
        if (status != .ok and result_size != 0) {
            return error.InvalidRequest;
        }

        switch (status) {
            .not_found => if (operation == .get) return null else {
                return error.InvalidRequest;
            },
            .invalid_request => return error.InvalidRequest,
            .resource_exhausted => return error.ResourceExhausted,
            .unsupported_version => return error.UnsupportedVersion,
            .internal => return error.Internal,
            .ok => {},
        }
        if (operation == .put and result_size != 0) {
            return error.InvalidRequest;
        }
        const result_allocator = allocator orelse {
            return @constCast(&[_]u8{});
        };
        const result = result_allocator.alloc(u8, result_size) catch {
            return error.OutOfMemory;
        };
        errdefer result_allocator.free(result);
        reader.readSliceAll(result) catch {
            return readerError(stream_reader.err);
        };
        return result;
    }

    fn validateRequest(
        self: *const Client,
        key: []const u8,
        value: []const u8,
    ) Error!void {
        if (self.namespace.len == 0 or
            self.namespace.len > std.math.maxInt(u16))
        {
            return error.InvalidConfiguration;
        }
        if (key.len == 0 or key.len > self.options.max_key_bytes or
            key.len > std.math.maxInt(u32) or
            value.len > self.options.max_value_bytes or
            value.len > std.math.maxInt(u32))
        {
            return error.ResourceExhausted;
        }
    }

    fn waitToRetry(
        self: *const Client,
        deadline: std.Io.Timestamp,
    ) Error!void {
        const now = std.Io.Clock.awake.now(self.io);
        if (now.nanoseconds >= deadline.nanoseconds) {
            return error.DeadlineExceeded;
        }
        const remaining = now.durationTo(deadline);
        const delay = std.Io.Duration.fromNanoseconds(@min(
            remaining.nanoseconds,
            self.options.retry_delay.nanoseconds,
        ));
        self.io.sleep(delay, .awake) catch |err| {
            return transportError(err);
        };
    }
};

const Key = struct {
    namespace: []const u8,
    key: []const u8,
};

const KeyContext = struct {
    pub fn hash(_: KeyContext, key: Key) u64 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(key.namespace);
        hasher.update(key.key);
        return hasher.final();
    }

    pub fn eql(_: KeyContext, left: Key, right: Key) bool {
        return std.mem.eql(u8, left.namespace, right.namespace) and
            std.mem.eql(u8, left.key, right.key);
    }
};

const Values = std.HashMap(
    Key,
    []u8,
    KeyContext,
    std.hash_map.default_max_load_percentage,
);

pub const Server = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    listener: std.Io.net.Server,
    options: Options,
    values: Values,
    values_mutex: std.Io.Mutex = .init,
    error_mutex: std.Io.Mutex = .init,
    run_error: ?Error = null,
    accept_group: std.Io.Group = .init,
    connection_group: std.Io.Group = .init,
    connection_slots: std.Io.Semaphore,
    started: std.atomic.Value(bool) = .init(false),
    stopped: std.atomic.Value(bool) = .init(false),

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        bind_address: std.Io.net.IpAddress,
        options: Options,
    ) Error!Server {
        const listener = bind_address.listen(io, .{
            .reuse_address = true,
        }) catch |err| return transportError(err);
        return .{
            .allocator = allocator,
            .io = io,
            .listener = listener,
            .options = options,
            .values = .init(allocator),
            .connection_slots = .{ .permits = options.max_connections },
        };
    }

    pub fn start(self: *Server) Error!void {
        if (self.started.swap(true, .acq_rel)) {
            return error.InvalidLifecycle;
        }
        self.accept_group.concurrent(
            self.io,
            acceptLoop,
            .{self},
        ) catch {
            self.started.store(false, .release);
            return error.Unavailable;
        };
    }

    pub fn stop(self: *Server) void {
        if (!self.started.load(.acquire)) return;
        if (self.stopped.swap(true, .acq_rel)) return;
        self.accept_group.cancel(self.io);
        self.connection_group.cancel(self.io);
    }

    pub fn deinit(self: *Server) void {
        self.stop();
        var iterator = self.values.iterator();
        while (iterator.next()) |entry| {
            self.allocator.free(entry.key_ptr.namespace);
            self.allocator.free(entry.key_ptr.key);
            self.allocator.free(entry.value_ptr.*);
        }
        self.values.deinit();
        self.listener.deinit(self.io);
    }

    pub fn address(self: *const Server) std.Io.net.IpAddress {
        return self.listener.socket.address;
    }

    pub fn check(self: *Server) Error!void {
        self.error_mutex.lockUncancelable(self.io);
        defer self.error_mutex.unlock(self.io);
        if (self.run_error) |err| return err;
    }

    fn acceptLoop(self: *Server) std.Io.Cancelable!void {
        while (true) {
            const stream = self.listener.accept(self.io) catch |err| {
                switch (err) {
                    error.Canceled, error.SocketNotListening => return,
                    else => {
                        self.recordError(error.Unavailable);
                        return;
                    },
                }
            };
            self.connection_group.concurrent(
                self.io,
                handleConnection,
                .{ self, stream },
            ) catch {
                self.handleConnection(stream) catch return;
            };
        }
    }

    fn handleConnection(
        self: *Server,
        stream: std.Io.net.Stream,
    ) std.Io.Cancelable!void {
        defer stream.close(self.io);
        try self.connection_slots.wait(self.io);
        defer self.connection_slots.post(self.io);
        self.handle(stream) catch |err| {
            log.debug("connection error={s}", .{@errorName(err)});
        };
    }

    fn handle(self: *Server, stream: std.Io.net.Stream) !void {
        var read_buffer: [1024]u8 = undefined;
        var stream_reader = stream.reader(self.io, &read_buffer);
        const reader = &stream_reader.interface;
        var write_buffer: [1024]u8 = undefined;
        var stream_writer = stream.writer(self.io, &write_buffer);
        const writer = &stream_writer.interface;

        var request_magic: [magic.len]u8 = undefined;
        reader.readSliceAll(&request_magic) catch return;
        if (!std.mem.eql(u8, magic, &request_magic)) {
            try writeResponse(writer, .invalid_request, "");
            return;
        }
        const version = reader.takeInt(u16, .big) catch return;
        if (version != protocol_version) {
            try writeResponse(writer, .unsupported_version, "");
            return;
        }
        const operation_int = reader.takeInt(u8, .big) catch return;
        const reserved = reader.takeInt(u8, .big) catch return;
        const namespace_size = reader.takeInt(u16, .big) catch return;
        const key_size = reader.takeInt(u32, .big) catch return;
        const value_size = reader.takeInt(u32, .big) catch return;
        const operation = std.enums.fromInt(
            Operation,
            operation_int,
        ) orelse {
            try writeResponse(writer, .invalid_request, "");
            return;
        };
        if (reserved != 0 or namespace_size == 0 or key_size == 0 or
            (operation == .get and value_size != 0))
        {
            try writeResponse(writer, .invalid_request, "");
            return;
        }
        if (namespace_size > self.options.max_key_bytes or
            key_size > self.options.max_key_bytes or
            value_size > self.options.max_value_bytes)
        {
            try writeResponse(writer, .resource_exhausted, "");
            return;
        }

        const namespace = self.allocator.alloc(
            u8,
            namespace_size,
        ) catch {
            try writeResponse(writer, .resource_exhausted, "");
            return;
        };
        defer self.allocator.free(namespace);
        reader.readSliceAll(namespace) catch return;

        const key = self.allocator.alloc(u8, key_size) catch {
            try writeResponse(writer, .resource_exhausted, "");
            return;
        };
        defer self.allocator.free(key);
        reader.readSliceAll(key) catch return;

        const value = self.allocator.alloc(u8, value_size) catch {
            try writeResponse(writer, .resource_exhausted, "");
            return;
        };
        defer self.allocator.free(value);
        reader.readSliceAll(value) catch return;

        switch (operation) {
            .get => {
                const result = self.get(namespace, key) catch {
                    try writeResponse(writer, .resource_exhausted, "");
                    return;
                };
                defer if (result) |bytes| self.allocator.free(bytes);
                if (result) |bytes| {
                    try writeResponse(writer, .ok, bytes);
                } else {
                    try writeResponse(writer, .not_found, "");
                }
            },
            .put => {
                self.put(namespace, key, value) catch {
                    try writeResponse(writer, .resource_exhausted, "");
                    return;
                };
                try writeResponse(writer, .ok, "");
            },
        }
    }

    fn get(
        self: *Server,
        namespace: []const u8,
        key: []const u8,
    ) std.mem.Allocator.Error!?[]u8 {
        self.values_mutex.lockUncancelable(self.io);
        defer self.values_mutex.unlock(self.io);
        const value = self.values.get(.{
            .namespace = namespace,
            .key = key,
        }) orelse return null;
        return try self.allocator.dupe(u8, value);
    }

    fn put(
        self: *Server,
        namespace: []const u8,
        key: []const u8,
        value: []const u8,
    ) std.mem.Allocator.Error!void {
        var stored_key: Key = .{
            .namespace = try self.allocator.dupe(u8, namespace),
            .key = undefined,
        };
        errdefer self.allocator.free(stored_key.namespace);
        stored_key.key = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(stored_key.key);
        const stored_value = try self.allocator.dupe(u8, value);
        errdefer self.allocator.free(stored_value);

        self.values_mutex.lockUncancelable(self.io);
        defer self.values_mutex.unlock(self.io);
        const result = try self.values.getOrPut(stored_key);
        if (result.found_existing) {
            self.allocator.free(stored_key.namespace);
            self.allocator.free(stored_key.key);
            self.allocator.free(result.value_ptr.*);
        }
        result.value_ptr.* = stored_value;
    }

    fn recordError(self: *Server, err: Error) void {
        self.error_mutex.lockUncancelable(self.io);
        defer self.error_mutex.unlock(self.io);
        if (self.run_error == null) self.run_error = err;
    }
};

fn writeResponse(
    writer: *std.Io.Writer,
    status: Status,
    value: []const u8,
) !void {
    try writer.writeAll(magic);
    try writer.writeInt(u16, protocol_version, .big);
    try writer.writeInt(u8, @intFromEnum(status), .big);
    try writer.writeInt(u8, 0, .big);
    try writer.writeInt(u32, @intCast(value.len), .big);
    try writer.writeAll(value);
    try writer.flush();
}

fn deadlineFromNow(
    io: std.Io,
    duration: std.Io.Duration,
) std.Io.Timestamp {
    return std.Io.Clock.awake.now(io).addDuration(duration);
}

fn deadlineReached(io: std.Io, deadline: std.Io.Timestamp) bool {
    return std.Io.Clock.awake.now(io).nanoseconds >= deadline.nanoseconds;
}

fn waitUntil(
    io: std.Io,
    deadline: std.Io.Timestamp,
) std.Io.Cancelable!void {
    return deadline.withClock(.awake).wait(io);
}

fn cancelRequest(
    select: *std.Io.Select(Selection),
    allocator: ?std.mem.Allocator,
) void {
    while (select.cancel()) |selected| switch (selected) {
        .timeout => {},
        .request => |result| {
            const value = result catch continue;
            if (value) |bytes| {
                if (allocator) |result_allocator| {
                    result_allocator.free(bytes);
                }
            }
        },
    };
}

fn transportError(err: anyerror) Error {
    return switch (err) {
        error.Canceled => error.Canceled,
        else => error.Unavailable,
    };
}

fn readerError(err: ?std.Io.net.Stream.Reader.Error) Error {
    return if (err) |value| transportError(value) else error.InvalidRequest;
}

fn writerError(err: ?std.Io.net.Stream.Writer.Error) Error {
    return if (err) |value| transportError(value) else error.Unavailable;
}

fn pjrtError(err: Error) pjrt.KeyValueStoreError {
    return switch (err) {
        error.DeadlineExceeded => error.DeadlineExceeded,
        error.OutOfMemory, error.ResourceExhausted => error.OutOfMemory,
        error.Canceled, error.Unavailable => error.Unavailable,
        else => error.Internal,
    };
}

const test_options: Options = .{
    .startup_timeout = .fromSeconds(1),
    .operation_timeout = .fromSeconds(1),
    .retry_delay = .fromMilliseconds(2),
    .max_key_bytes = 64,
    .max_value_bytes = 64,
    .max_connections = 8,
};

test "key-value store operations, namespaces, limits, and stop" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var server = try Server.init(
        allocator,
        io,
        .{ .ip4 = .loopback(0) },
        test_options,
    );
    defer server.deinit();
    try server.start();

    const client_a: Client = .init(
        io,
        server.address(),
        "job-a",
        test_options,
    );
    const client_b: Client = .init(
        io,
        server.address(),
        "job-b",
        test_options,
    );

    try client_a.put("key", "first");
    var value = (try client_a.tryGet(allocator, "key")).?;
    try std.testing.expectEqualStrings("first", value);
    allocator.free(value);

    try client_a.put("key", "second");
    value = try client_a.get(
        allocator,
        "key",
        .fromMilliseconds(100),
    );
    try std.testing.expectEqualStrings("second", value);
    allocator.free(value);
    try std.testing.expectEqual(
        null,
        try client_b.tryGet(allocator, "key"),
    );

    try client_a.put("empty", "");
    value = try client_a.get(
        allocator,
        "empty",
        .fromMilliseconds(100),
    );
    try std.testing.expectEqual(0, value.len);
    allocator.free(value);

    var max_key: [test_options.max_key_bytes]u8 = undefined;
    @memset(&max_key, 'k');
    var max_value: [test_options.max_value_bytes]u8 = undefined;
    @memset(&max_value, 'v');
    try client_a.put(&max_key, &max_value);
    try std.testing.expectError(
        error.ResourceExhausted,
        client_a.put("x", &([_]u8{'v'} * *65)),
    );
    try std.testing.expectError(
        error.DeadlineExceeded,
        client_a.get(
            allocator,
            "missing",
            .fromMilliseconds(20),
        ),
    );

    try expectRawStatus(
        io,
        server.address(),
        "BAD!",
        protocol_version,
        @intFromEnum(Operation.get),
        1,
        1,
        0,
        .invalid_request,
    );
    try expectRawStatus(
        io,
        server.address(),
        magic,
        protocol_version + 1,
        @intFromEnum(Operation.get),
        1,
        1,
        0,
        .unsupported_version,
    );
    try expectRawStatus(
        io,
        server.address(),
        magic,
        protocol_version,
        @intFromEnum(Operation.get),
        1,
        test_options.max_key_bytes + 1,
        0,
        .resource_exhausted,
    );

    const partial_stream = try server.address().connect(io, .{
        .mode = .stream,
    });
    defer partial_stream.close(io);
    var buffer: [1]u8 = undefined;
    var writer = partial_stream.writer(io, &buffer);
    try writer.interface.writeByte('Z');
    try writer.interface.flush();
    try io.sleep(.fromMilliseconds(10), .awake);
    server.stop();
    server.stop();
    try server.check();
}

test "key-value client retries until the server starts" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const loopback: std.Io.net.IpAddress = .{ .ip4 = .loopback(0) };
    var reservation = try loopback.listen(io, .{
        .reuse_address = true,
    });
    const address = reservation.socket.address;
    reservation.deinit(io);

    const client: Client = .init(io, address, "retry", test_options);
    var future = try io.concurrent(putForTest, .{&client});
    try io.sleep(.fromMilliseconds(20), .awake);

    var server = try Server.init(
        allocator,
        io,
        address,
        test_options,
    );
    defer server.deinit();
    try server.start();
    try future.await(io);

    const value = (try client.tryGet(allocator, "ready")).?;
    defer allocator.free(value);
    try std.testing.expectEqualStrings("yes", value);
}

test "key-value server bounds concurrent handlers" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var server = try Server.init(
        allocator,
        io,
        .{ .ip4 = .loopback(0) },
        test_options,
    );
    defer server.deinit();
    try server.start();
    const client: Client = .init(
        io,
        server.address(),
        "concurrent",
        test_options,
    );

    var failed: std.atomic.Value(bool) = .init(false);
    var group: std.Io.Group = .init;
    for (0..64) |index| {
        try group.concurrent(
            io,
            concurrentClient,
            .{ &client, allocator, index, &failed },
        );
    }
    try group.await(io);
    try std.testing.expect(!failed.load(.acquire));
}

fn putForTest(client: *const Client) Error!void {
    return client.put("ready", "yes");
}

fn concurrentClient(
    client: *const Client,
    allocator: std.mem.Allocator,
    index: usize,
    failed: *std.atomic.Value(bool),
) void {
    var key_buffer: [32]u8 = undefined;
    const key = std.fmt.bufPrint(&key_buffer, "key-{d}", .{index}) catch {
        failed.store(true, .release);
        return;
    };
    client.put(key, "value") catch {
        failed.store(true, .release);
        return;
    };
    const value = client.get(
        allocator,
        key,
        .fromSeconds(1),
    ) catch {
        failed.store(true, .release);
        return;
    };
    defer allocator.free(value);
    if (!std.mem.eql(u8, value, "value")) {
        failed.store(true, .release);
    }
}

fn expectRawStatus(
    io: std.Io,
    address: std.Io.net.IpAddress,
    request_magic: []const u8,
    version: u16,
    operation: u8,
    namespace_size: usize,
    key_size: usize,
    value_size: usize,
    expected: Status,
) !void {
    const stream = try address.connect(io, .{ .mode = .stream });
    defer stream.close(io);
    var write_buffer: [128]u8 = undefined;
    var stream_writer = stream.writer(io, &write_buffer);
    const writer = &stream_writer.interface;
    try writer.writeAll(request_magic);
    try writer.writeInt(u16, version, .big);
    try writer.writeInt(u8, operation, .big);
    try writer.writeInt(u8, 0, .big);
    try writer.writeInt(u16, @intCast(namespace_size), .big);
    try writer.writeInt(u32, @intCast(key_size), .big);
    try writer.writeInt(u32, @intCast(value_size), .big);
    try writer.flush();

    var read_buffer: [128]u8 = undefined;
    var stream_reader = stream.reader(io, &read_buffer);
    const reader = &stream_reader.interface;
    var response_magic: [magic.len]u8 = undefined;
    try reader.readSliceAll(&response_magic);
    try std.testing.expectEqualStrings(magic, &response_magic);
    try std.testing.expectEqual(
        protocol_version,
        try reader.takeInt(u16, .big),
    );
    try std.testing.expectEqual(
        @intFromEnum(expected),
        try reader.takeInt(u8, .big),
    );
    try std.testing.expectEqual(0, try reader.takeInt(u8, .big));
    try std.testing.expectEqual(0, try reader.takeInt(u32, .big));
}
