const std = @import("std");

const pjrt = @import("pjrt");
const kv_store = @import("distributed/kv_store.zig");

pub const Error = kv_store.Error;

/// Fixed-size distributed process configuration.
///
/// Every process in one job uses the same coordinator, process count, and
/// namespace. The namespace should be unique for each launcher invocation.
pub const Config = struct {
    coordinator_address: std.Io.net.IpAddress,
    process_index: usize,
    process_count: usize,
    namespace: []const u8,
    local_device_ids: []const i64 = &.{},
    bind_address: ?std.Io.net.IpAddress = null,
    startup_timeout: std.Io.Duration = .fromSeconds(120),
    operation_timeout: std.Io.Duration = .fromSeconds(30),
    shutdown_timeout: std.Io.Duration = .fromSeconds(30),
    retry_delay: std.Io.Duration = .fromMilliseconds(50),
    max_key_bytes: usize = 64 * 1024,
    max_value_bytes: usize = 16 * 1024 * 1024,
    max_connections: usize = 64,
};

/// Move-stable owner of the distributed control plane.
///
/// Process 0 owns the server; every process owns an immutable client. The
/// allocated state keeps PJRT callback pointers valid even if `Runtime` moves.
pub const Runtime = struct {
    state: ?*State,

    const State = struct {
        allocator: std.mem.Allocator,
        config: Config,
        client: kv_store.Client,
        key_value_store: pjrt.KeyValueStore,
        server: ?kv_store.Server,
        barrier_generation: std.atomic.Value(u64) = .init(0),
        destroying_client: std.atomic.Value(bool) = .init(false),
    };

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        config: Config,
    ) Error!Runtime {
        try validateConfig(config);

        const state = allocator.create(State) catch {
            return error.OutOfMemory;
        };
        errdefer allocator.destroy(state);

        const namespace = allocator.dupe(u8, config.namespace) catch {
            return error.OutOfMemory;
        };
        errdefer allocator.free(namespace);
        const local_device_ids = allocator.dupe(
            i64,
            config.local_device_ids,
        ) catch return error.OutOfMemory;
        errdefer allocator.free(local_device_ids);

        var owned_config = config;
        owned_config.namespace = namespace;
        owned_config.local_device_ids = local_device_ids;
        const options = optionsFromConfig(owned_config);
        state.* = .{
            .allocator = allocator,
            .config = owned_config,
            .client = .init(
                io,
                owned_config.coordinator_address,
                owned_config.namespace,
                options,
            ),
            .key_value_store = undefined,
            .server = null,
        };
        state.key_value_store = state.client.keyValueStore();

        if (owned_config.process_index == 0) {
            state.server = try .init(
                allocator,
                io,
                bindAddress(owned_config),
                options,
            );
            errdefer if (state.server) |*server| server.deinit();
            try state.server.?.start();
        }
        return .{ .state = state };
    }

    /// Stops the locally owned server and frees copied configuration.
    pub fn deinit(self: *Runtime) void {
        const state = self.state orelse return;
        self.state = null;
        if (state.server) |*server| server.deinit();
        state.allocator.free(state.config.local_device_ids);
        state.allocator.free(state.config.namespace);
        state.allocator.destroy(state);
    }

    /// Returns a stable callback table that must outlive the PJRT client.
    pub fn keyValueStore(
        self: *Runtime,
    ) Error!*const pjrt.KeyValueStore {
        const state = self.state orelse return error.InvalidLifecycle;
        return &state.key_value_store;
    }

    /// Synchronizes every process at a reusable, generation-safe name.
    pub fn barrier(self: *Runtime, name: []const u8) Error!void {
        const state = self.state orelse return error.InvalidLifecycle;
        return barrierUntil(
            state,
            name,
            deadlineFromNow(
                state.client.io,
                state.config.operation_timeout,
            ),
        );
    }

    /// Destroys follower PJRT clients before process 0 destroys its client.
    ///
    /// The separate key-value client remains alive after follower destruction,
    /// so followers can acknowledge completion while process 0 still services
    /// PJRT coordination requests.
    pub fn destroyPjrtClient(
        self: *Runtime,
        api: *const pjrt.Api,
        client: *?*pjrt.Client,
    ) Error!void {
        const state = self.state orelse return error.InvalidLifecycle;
        const pjrt_client = client.* orelse return error.InvalidLifecycle;
        if (state.destroying_client.swap(true, .acq_rel)) {
            return error.InvalidLifecycle;
        }
        client.* = null;
        const deadline = deadlineFromNow(
            state.client.io,
            state.config.shutdown_timeout,
        );
        barrierUntil(state, "shutdown-ready", deadline) catch |err| {
            pjrt_client.deinit(api);
            return err;
        };

        if (state.config.process_index != 0) {
            pjrt_client.deinit(api);
            return acknowledgeClientDestroyed(state, deadline);
        }
        waitForFollowers(state, deadline) catch |err| {
            pjrt_client.deinit(api);
            return err;
        };
        pjrt_client.deinit(api);
    }
};

fn validateConfig(config: Config) Error!void {
    if (config.process_count == 0 or
        config.process_index >= config.process_count or
        config.namespace.len == 0 or
        config.namespace.len > std.math.maxInt(u16) or
        config.namespace.len > config.max_key_bytes or
        config.max_key_bytes == 0 or
        config.max_key_bytes > std.math.maxInt(u32) or
        config.max_value_bytes == 0 or
        config.max_value_bytes > std.math.maxInt(u32) or
        config.max_connections == 0 or
        config.startup_timeout.nanoseconds <= 0 or
        config.operation_timeout.nanoseconds <= 0 or
        config.shutdown_timeout.nanoseconds <= 0 or
        config.retry_delay.nanoseconds <= 0)
    {
        return error.InvalidConfiguration;
    }
    if (config.bind_address) |address| {
        if (config.process_index != 0 or
            address.getPort() != config.coordinator_address.getPort())
        {
            return error.InvalidConfiguration;
        }
    }
}

fn optionsFromConfig(config: Config) kv_store.Options {
    return .{
        .startup_timeout = config.startup_timeout,
        .operation_timeout = config.operation_timeout,
        .retry_delay = config.retry_delay,
        .max_key_bytes = config.max_key_bytes,
        .max_value_bytes = config.max_value_bytes,
        .max_connections = config.max_connections,
    };
}

fn bindAddress(config: Config) std.Io.net.IpAddress {
    return config.bind_address orelse switch (config.coordinator_address) {
        .ip4 => |address| .{ .ip4 = .unspecified(address.port) },
        .ip6 => |address| .{ .ip6 = .unspecified(address.port) },
    };
}

fn barrierUntil(
    state: *Runtime.State,
    name: []const u8,
    deadline: std.Io.Timestamp,
) Error!void {
    if (name.len == 0) return error.InvalidRequest;
    if (state.server) |*server| try server.check();
    const generation = state.barrier_generation.fetchAdd(1, .seq_cst);
    const prefix = std.fmt.allocPrint(
        state.allocator,
        "barrier/{d}/{d}/{s}",
        .{ generation, name.len, name },
    ) catch return error.OutOfMemory;
    defer state.allocator.free(prefix);

    const own_key = try processKey(
        state.allocator,
        prefix,
        state.config.process_index,
    );
    defer state.allocator.free(own_key);
    try state.client.putUntil(own_key, "", deadline);

    for (0..state.config.process_count) |process_index| {
        const key = try processKey(
            state.allocator,
            prefix,
            process_index,
        );
        defer state.allocator.free(key);
        const value = try state.client.getUntil(
            state.allocator,
            key,
            deadline,
        );
        state.allocator.free(value);
    }
}

fn acknowledgeClientDestroyed(
    state: *Runtime.State,
    deadline: std.Io.Timestamp,
) Error!void {
    const key = try processKey(
        state.allocator,
        "shutdown/client-destroyed",
        state.config.process_index,
    );
    defer state.allocator.free(key);
    return state.client.putUntil(key, "", deadline);
}

fn waitForFollowers(
    state: *Runtime.State,
    deadline: std.Io.Timestamp,
) Error!void {
    for (1..state.config.process_count) |process_index| {
        const key = try processKey(
            state.allocator,
            "shutdown/client-destroyed",
            process_index,
        );
        defer state.allocator.free(key);
        const value = try state.client.getUntil(
            state.allocator,
            key,
            deadline,
        );
        state.allocator.free(value);
    }
}

fn processKey(
    allocator: std.mem.Allocator,
    prefix: []const u8,
    process_index: usize,
) Error![]u8 {
    return std.fmt.allocPrint(
        allocator,
        "{s}/{d}",
        .{ prefix, process_index },
    ) catch error.OutOfMemory;
}

fn deadlineFromNow(
    io: std.Io,
    duration: std.Io.Duration,
) std.Io.Timestamp {
    return std.Io.Clock.awake.now(io).addDuration(duration);
}

test "distributed config validation" {
    const address: std.Io.net.IpAddress = .{
        .ip4 = .loopback(8910),
    };
    const valid: Config = .{
        .coordinator_address = address,
        .process_index = 0,
        .process_count = 2,
        .namespace = "test",
    };
    try validateConfig(valid);

    var invalid = valid;
    invalid.process_index = 2;
    try std.testing.expectError(
        error.InvalidConfiguration,
        validateConfig(invalid),
    );
    invalid = valid;
    invalid.namespace = "";
    try std.testing.expectError(
        error.InvalidConfiguration,
        validateConfig(invalid),
    );
}

test "distributed runtime supports repeated named barriers" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const loopback: std.Io.net.IpAddress = .{ .ip4 = .loopback(0) };
    var reservation = try loopback.listen(io, .{
        .reuse_address = true,
    });
    const address = reservation.socket.address;
    reservation.deinit(io);

    const common: Config = .{
        .coordinator_address = address,
        .process_index = 0,
        .process_count = 2,
        .namespace = "runtime-test",
        .operation_timeout = .fromSeconds(1),
        .retry_delay = .fromMilliseconds(2),
    };
    var rank0 = try Runtime.init(allocator, io, common);
    defer rank0.deinit();
    var rank1_config = common;
    rank1_config.process_index = 1;
    var rank1 = try Runtime.init(allocator, io, rank1_config);
    defer rank1.deinit();

    for (0..2) |_| {
        var rank0_barrier = try io.concurrent(
            barrierForTest,
            .{&rank0},
        );
        try rank1.barrier("same-name");
        try rank0_barrier.await(io);
    }

    rank1.deinit();
    rank0.deinit();
}

fn barrierForTest(runtime: *Runtime) Error!void {
    return runtime.barrier("same-name");
}
