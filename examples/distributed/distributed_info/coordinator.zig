const std = @import("std");

const pjrt = @import("pjrt");

const max_payload_size = 16 * 1024 * 1024;
const retry_delay_ms = 50;

const Operation = enum(u8) {
    get = 1,
    put = 2,
    shutdown = 3,
};

const Status = enum(u8) {
    ok = 0,
    not_found = 1,
    failed = 2,
};

pub const Client = struct {
    io: std.Io,
    address: std.Io.net.IpAddress,

    pub fn keyValueStore(self: *Client) pjrt.KeyValueStore {
        return .{
            .user_arg = self,
            .get = getCallback,
            .try_get = tryGetCallback,
            .put = putCallback,
        };
    }

    pub fn barrier(
        self: *Client,
        allocator: std.mem.Allocator,
        rank: usize,
        process_count: usize,
    ) pjrt.KeyValueStoreError!void {
        var key_buffer: [64]u8 = undefined;
        const own_key = std.fmt.bufPrint(
            &key_buffer,
            "distributed_info/done/{d}",
            .{rank},
        ) catch return error.Internal;
        try self.putWithRetry(own_key, "done");

        for (0..process_count) |process_index| {
            const key = std.fmt.bufPrint(
                &key_buffer,
                "distributed_info/done/{d}",
                .{process_index},
            ) catch return error.Internal;
            const value = try self.get(allocator, key, 30_000);
            allocator.free(value);
        }
    }

    pub fn shutdown(self: *Client) pjrt.KeyValueStoreError!void {
        _ = try self.request(.shutdown, "", "", null);
    }

    fn getCallback(
        user_arg: *anyopaque,
        allocator: std.mem.Allocator,
        key: []const u8,
        timeout_ms: c_int,
    ) pjrt.KeyValueStoreError![]u8 {
        const self: *Client = @ptrCast(@alignCast(user_arg));
        const timeout: usize = if (timeout_ms <= 0)
            30_000
        else
            @intCast(timeout_ms);
        return self.get(allocator, key, timeout);
    }

    fn tryGetCallback(
        user_arg: *anyopaque,
        allocator: std.mem.Allocator,
        key: []const u8,
    ) pjrt.KeyValueStoreError!?[]u8 {
        const self: *Client = @ptrCast(@alignCast(user_arg));
        return self.request(.get, key, "", allocator);
    }

    fn putCallback(
        user_arg: *anyopaque,
        key: []const u8,
        value: []const u8,
    ) pjrt.KeyValueStoreError!void {
        const self: *Client = @ptrCast(@alignCast(user_arg));
        try self.putWithRetry(key, value);
    }

    fn get(
        self: *Client,
        allocator: std.mem.Allocator,
        key: []const u8,
        timeout_ms: usize,
    ) pjrt.KeyValueStoreError![]u8 {
        const attempts = @max(1, timeout_ms / retry_delay_ms);
        for (0..attempts) |_| {
            const result = self.request(.get, key, "", allocator) catch null;
            if (result) |value| return value;
            self.io.sleep(.fromMilliseconds(retry_delay_ms), .awake) catch {
                return error.Unavailable;
            };
        }
        return error.DeadlineExceeded;
    }

    fn putWithRetry(
        self: *Client,
        key: []const u8,
        value: []const u8,
    ) pjrt.KeyValueStoreError!void {
        for (0..600) |_| {
            _ = self.request(.put, key, value, null) catch {
                self.io.sleep(
                    .fromMilliseconds(retry_delay_ms),
                    .awake,
                ) catch return error.Unavailable;
                continue;
            };
            return;
        }
        return error.Unavailable;
    }

    fn request(
        self: *Client,
        operation: Operation,
        key: []const u8,
        value: []const u8,
        allocator: ?std.mem.Allocator,
    ) pjrt.KeyValueStoreError!?[]u8 {
        if (key.len > max_payload_size or value.len > max_payload_size) {
            return error.Internal;
        }

        const stream = self.address.connect(self.io, .{
            .mode = .stream,
        }) catch return error.Unavailable;
        defer stream.close(self.io);

        var write_buffer: [1024]u8 = undefined;
        var stream_writer = stream.writer(self.io, &write_buffer);
        const writer = &stream_writer.interface;
        writer.writeInt(u8, @intFromEnum(operation), .little) catch {
            return error.Unavailable;
        };
        writer.writeInt(u32, @intCast(key.len), .little) catch {
            return error.Unavailable;
        };
        writer.writeInt(u32, @intCast(value.len), .little) catch {
            return error.Unavailable;
        };
        writer.writeAll(key) catch return error.Unavailable;
        writer.writeAll(value) catch return error.Unavailable;
        writer.flush() catch return error.Unavailable;

        var read_buffer: [1024]u8 = undefined;
        var stream_reader = stream.reader(self.io, &read_buffer);
        const reader = &stream_reader.interface;
        const status_int = reader.takeInt(u8, .little) catch {
            return error.Unavailable;
        };
        const status: Status = std.enums.fromInt(Status, status_int) orelse {
            return error.Internal;
        };
        const result_size = reader.takeInt(u32, .little) catch {
            return error.Unavailable;
        };

        switch (status) {
            .not_found => return null,
            .failed => return error.Internal,
            .ok => {},
        }
        if (result_size == 0) return null;
        if (result_size > max_payload_size) return error.Internal;

        const result_allocator = allocator orelse return error.Internal;
        const result = result_allocator.alloc(u8, result_size) catch {
            return error.OutOfMemory;
        };
        errdefer result_allocator.free(result);
        reader.readSliceAll(result) catch return error.Unavailable;
        return result;
    }
};

pub const Server = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    listener: std.Io.net.Server,
    values: std.StringHashMap([]u8),
    run_error: ?anyerror = null,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        address: std.Io.net.IpAddress,
    ) !Server {
        const bind_address: std.Io.net.IpAddress = switch (address) {
            .ip4 => |value| .{
                .ip4 = .unspecified(value.port),
            },
            .ip6 => |value| .{
                .ip6 = .unspecified(value.port),
            },
        };
        return .{
            .allocator = allocator,
            .io = io,
            .listener = try bind_address.listen(io, .{
                .reuse_address = true,
            }),
            .values = .init(allocator),
        };
    }

    pub fn deinit(self: *Server) void {
        var iterator = self.values.iterator();
        while (iterator.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.values.deinit();
        self.listener.deinit(self.io);
    }

    pub fn runThread(self: *Server) void {
        self.run() catch |err| {
            self.run_error = err;
        };
    }

    fn run(self: *Server) !void {
        while (true) {
            const stream = try self.listener.accept(self.io);
            defer stream.close(self.io);
            if (try self.handle(stream)) return;
        }
    }

    fn handle(self: *Server, stream: std.Io.net.Stream) !bool {
        var read_buffer: [1024]u8 = undefined;
        var stream_reader = stream.reader(self.io, &read_buffer);
        const reader = &stream_reader.interface;

        const operation: Operation = std.enums.fromInt(
            Operation,
            try reader.takeInt(u8, .little),
        ) orelse return error.InvalidOperation;
        const key_size = try reader.takeInt(u32, .little);
        const value_size = try reader.takeInt(u32, .little);
        if (key_size > max_payload_size or value_size > max_payload_size) {
            return error.PayloadTooLarge;
        }

        const key = try self.allocator.alloc(u8, key_size);
        defer self.allocator.free(key);
        try reader.readSliceAll(key);

        const value = try self.allocator.alloc(u8, value_size);
        defer self.allocator.free(value);
        try reader.readSliceAll(value);

        var write_buffer: [1024]u8 = undefined;
        var stream_writer = stream.writer(self.io, &write_buffer);
        const writer = &stream_writer.interface;

        switch (operation) {
            .get => {
                if (self.values.get(key)) |stored| {
                    try writeResponse(writer, .ok, stored);
                } else {
                    try writeResponse(writer, .not_found, "");
                }
            },
            .put => {
                try self.put(key, value);
                try writeResponse(writer, .ok, "");
            },
            .shutdown => {
                try writeResponse(writer, .ok, "");
                return true;
            },
        }
        return false;
    }

    fn put(self: *Server, key: []const u8, value: []const u8) !void {
        const stored_value = try self.allocator.dupe(u8, value);
        errdefer self.allocator.free(stored_value);
        if (self.values.getPtr(key)) |existing| {
            self.allocator.free(existing.*);
            existing.* = stored_value;
            return;
        }

        const stored_key = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(stored_key);
        try self.values.put(stored_key, stored_value);
    }

    fn writeResponse(
        writer: *std.Io.Writer,
        status: Status,
        value: []const u8,
    ) !void {
        try writer.writeInt(u8, @intFromEnum(status), .little);
        try writer.writeInt(u32, @intCast(value.len), .little);
        try writer.writeAll(value);
        try writer.flush();
    }
};
