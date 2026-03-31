const std = @import("std");
const builtin = @import("builtin");

const stdx = @import("stdx");

const zml = @import("../zml.zig");

const log = std.log.scoped(.@"zml/attnd");

var context_type_id: ?zml.pjrt.ffi.TypeId = null;

const Context = struct { client: Client, allocator: std.mem.Allocator, io: std.Io };

pub const Config = struct {
    desctination: std.Io.net.IpAddress,
    client_options: Options = .{},
};

pub const Parameters = struct {
    model_id: ModelId,
    head_dim: u32,
    head_bytes: u32,
    num_kv_heads: u8,
    num_q_per_head: u8,
    num_q_per_packet: u8,

    pub const Init = struct {
        model_id: ModelId,
        head_dim: u32,
        num_attention_heads: u32,
        num_kv_heads: u8,
        mtu: u32,
    };

    pub fn init(args: Init) Parameters {
        const num_q_per_head: u8 = @intCast(@divExact(args.num_attention_heads, args.num_kv_heads));
        const head_bytes = args.head_dim * @sizeOf(u16); // TODO: Here we assume bf16 or equivalent.
        const kv_bytes = 2 * head_bytes;

        var max_q_per_packet: u8 = @intCast(std.math.divFloor(u32, args.mtu - @sizeOf(Header) - kv_bytes, head_bytes) catch unreachable);
        max_q_per_packet = @min(max_q_per_packet, num_q_per_head);

        const num_packets_per_head: u8 = @intCast(std.math.divCeil(u32, num_q_per_head, max_q_per_packet) catch unreachable);
        const num_q_per_packet = @divExact(num_q_per_head, num_packets_per_head);
        if (num_q_per_packet < num_q_per_head) {
            log.warn("Will split attnd requests in {d} packets of {d} queries", .{ num_packets_per_head, max_q_per_packet });
        }

        return .{
            .model_id = args.model_id,
            .head_dim = args.head_dim,
            .head_bytes = head_bytes,
            .num_kv_heads = args.num_kv_heads,
            .num_q_per_head = num_q_per_head,
            .num_q_per_packet = num_q_per_packet,
        };
    }
};

pub const Metadata = struct {
    pub fn initBuffer(
        self: Metadata,
        io: std.Io,
        platform: *const zml.Platform,
        sharding: zml.sharding.Sharding,
    ) !zml.Bufferized(Metadata) {
        _ = self; // autofix
        _ = io; // autofix
        _ = platform; // autofix
        _ = sharding; // autofix
    }

    pub fn deinitBuffer(self: *zml.Bufferized(Metadata)) void {
        _ = self; // autofix
    }
};

pub fn register(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, config: Config) !void {
    var ctx = try allocator.create(Context);
    errdefer allocator.destroy(ctx);

    ctx.allocator = allocator;
    ctx.io = io;
    ctx.client = try Client.init(io, config.desctination, config.client_options);
    errdefer ctx.client.deinit(io);

    if (context_type_id) |_| {
        @panic("Cannot configure the attnd context twice.");
    }

    context_type_id = try platform.registerData("attnd", ctx, &.{
        .deleter = deleter,
    });

    switch (platform.target) {
        .cpu => {
            try targets.cpu.register(platform);
        },
        else => @panic("Not supported yet"),
    }
}

fn deleter(object: ?*anyopaque) callconv(.c) void {
    const ptr = object.?;
    const ctx: *Context = @ptrCast(@alignCast(ptr));
    ctx.client.deinit(ctx.io);
}

pub fn causalAttention(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_pos: zml.Tensor, metadata: Metadata, parameters: Parameters) zml.Tensor {
    _ = metadata; // autofix
    // Here is doesn't matter which target we call, they all have the same input/output/parameters
    const out = targets.cpu.call(
        .{
            .q = q,
            .k = k,
            .v = v,
            .token_pos = token_pos,
        },
        .{
            .attn = q.shape(),
        },
        parameters,
    );
    return out.attn;
}

pub const targets = struct {
    pub const cpu = zml.ops.CustomCall(Input, Output, Parameters, cpuCall, .{
        .name = "attnd",
        .sharding_aware = false,
        .has_side_effect = false,
        .output_operand_aliases = .{ .attn = .q },
    });
};

const Input = struct {
    q: zml.Tensor,
    k: zml.Tensor,
    v: zml.Tensor,
    token_pos: zml.Tensor,
};

const Output = struct {
    attn: zml.Shape,
};

fn cpuCall(
    call_frame: *zml.pjrt.ffi.CallFrame,
    input: zml.pjrtx.TensorToCustomCallBuffer(Input),
    output: zml.pjrtx.ShapeToCustomCallBuffer(Output),
    parameters: Parameters,
) !?*zml.pjrt.ffi.Error {
    const ctx: *Context = @ptrCast(@alignCast(try call_frame.ctx.getContext(context_type_id.?, call_frame.api)));

    var buffer: []u8 = try ctx.allocator.alloc(u8, @sizeOf(Header) + parameters.num_kv_heads * (2 + parameters.num_q_per_head) * parameters.head_dim * @sizeOf(u16));
    defer ctx.allocator.free(buffer);

    const req: Request = b: {
        const header: *align(1) Header = std.mem.bytesAsValue(Header, buffer[0..@sizeOf(Header)]);
        const payload: []u8 = buffer[@sizeOf(Header)..buffer.len];
        const q: []u8 = payload[0 .. parameters.num_q_per_packet * parameters.head_bytes];
        const k: []u8 = payload[q.len..(q.len + parameters.head_bytes)];
        const v: []u8 = payload[(q.len + parameters.head_bytes)..(q.len + 2 * parameters.head_bytes)];
        break :b .{
            .bytes = buffer,
            .header = header,
            .q = q,
            .k = k,
            .v = v,
        };
    };
    req.header.* = .{
        .type = .attn,
        .kv_head_id = undefined, // set in the for loop
        .first_q_id = undefined, // set in the inner while loop
        .num_queries = undefined, // set in the inner while loop
        .model_id = parameters.model_id,
        .conversation_id = 0,
        .layer_id = 0,
        .token_pos = std.mem.bytesAsValue(u32, asConstSlice(input.token_pos)[0..@sizeOf(u32)]).*,
    };

    const q = asConstSlice(input.q);
    const k = asConstSlice(input.k);
    const v = asConstSlice(input.v);
    const head_bytes = parameters.head_bytes;
    const num_queries = parameters.num_q_per_head;
    for (0..parameters.num_kv_heads) |kv_head_id| {
        const k_offset = kv_head_id * head_bytes;
        @memcpy(req.k, k[k_offset..][0..head_bytes]);

        const v_offset = kv_head_id * head_bytes;
        @memcpy(req.v, v[v_offset..][0..head_bytes]);

        req.header.kv_head_id = @intCast(kv_head_id);

        var q_offset = kv_head_id * head_bytes * num_queries;
        var q_sent: u8 = 0;

        while (num_queries - q_sent > 0) {
            const nq = @min(num_queries - q_sent, parameters.num_q_per_packet);
            @memcpy(req.q, q[q_offset..][0 .. head_bytes * nq]);
            q_offset += head_bytes * nq;

            req.header.num_queries = nq;
            req.header.first_q_id = q_sent;
            q_sent += nq;

            try ctx.client.send(ctx.io, req);
        }
    }

    //
    // ---------
    //

    const attn: []u8 = asSlice(output.attn);
    var recv_q: usize = 0;

    while (recv_q < parameters.num_kv_heads * parameters.num_q_per_head) {
        const resp = ctx.client.receive(ctx.io, buffer) catch |err| {
            log.err("Failed to receveive response: {any}", .{err});
            return err;
        };

        std.debug.assert(resp.header.num_queries <= parameters.num_q_per_head);
        std.debug.assert(resp.payload.len == resp.header.num_queries * parameters.head_bytes);
        std.debug.assert(resp.header.kv_head_id < parameters.num_kv_heads);
        std.debug.assert(resp.header.first_q_id + resp.header.num_queries <= parameters.num_q_per_head);

        const q_offset = (@as(usize, resp.header.kv_head_id) * parameters.num_q_per_head + resp.header.first_q_id) * parameters.head_bytes;
        @memcpy(attn[q_offset .. q_offset + resp.payload.len], resp.payload);
        recv_q += resp.header.num_queries;
    }

    return null;
}

fn asSlice(buf: zml.pjrtx.CustomCallBuffer) []u8 {
    const bytes: [*]u8 = @ptrCast(@alignCast(buf.ptr));
    return bytes[0..buf.shape.byteSize()];
}

fn asConstSlice(buf: zml.pjrtx.CustomCallBuffer) []const u8 {
    const bytes: [*]const u8 = @ptrCast(@alignCast(buf.ptr));
    return bytes[0..buf.shape.byteSize()];
}

pub const Options = struct {
    buffer_size: usize = 1500 * 64,
};

const Client = struct {
    socket: std.Io.net.Socket,
    destination: std.Io.net.IpAddress,

    fn init(io: std.Io, destination: std.Io.net.IpAddress, options: Options) !Client {
        const src: std.Io.net.IpAddress = .{ .ip4 = .unspecified(0) };
        const socket = try src.bind(io, .{ .mode = .dgram, .protocol = .udp });

        try std.posix.setsockopt(socket.handle, std.posix.SOL.SOCKET, std.posix.SO.RCVBUF, @ptrCast(&options.buffer_size));
        try std.posix.setsockopt(socket.handle, std.posix.SOL.SOCKET, std.posix.SO.SNDBUF, @ptrCast(&options.buffer_size));

        // Set the Don't Fragment bit to avoid fragmentation on IPv4
        switch (builtin.os.tag) {
            .linux => {
                // With PMTUDISC_DO the kernel will keep track of the ICMP messages to detect the real MTU.
                // Best practice [1] seems to be to use IP_PMTUDISC_PROBE and do DPLPMTUD to avoid Blind Performance-Degrading Attack [2]
                // But we assume to be within an internal network, otherwise attnd is useless anyway.
                // [1] https://seemann.io/posts/2025-02-19---ip-fragmentation/
                // [2] https://www.rfc-editor.org/rfc/rfc5927#section-7
                // TODO: Do we actually try to use the max MTU? If we just assume it to be fine, we could just as well use IP_PMTUDISC_PROBE
                const mtu_discover: u32 = std.os.linux.IP.PMTUDISC_DO;
                try std.posix.setsockopt(socket.handle, std.posix.IPPROTO.IP, std.os.linux.IP.MTU_DISCOVER, @ptrCast(&mtu_discover));
            },
            .macos => {
                // We could set IP_DONTFRAG on the socket, but Zig doesn't expose this option (yet).
            },
            else => {},
        }

        return .{
            .socket = socket,
            .destination = destination,
        };
    }

    pub fn send(self: *Client, io: std.Io, request: Request) !void {
        try self.socket.send(io, &self.destination, request.bytes);
    }

    pub fn receive(self: *Client, io: std.Io, buffer: []u8) !Response {
        const incoming = try self.socket.receive(io, buffer);
        return Response.fromBytes(incoming.data);
    }

    pub fn deinit(self: *Client, io: std.Io) void {
        self.socket.close(io);
    }
};

const Message = struct {
    bytes: []u8,
    header: *align(1) Header,
    payload: []u8,

    fn fromBytes(bytes: []u8) !Message {
        const n = @sizeOf(Header);
        if (bytes.len < n) return error.ShortMessage;

        const msg: Message = .{
            .bytes = bytes,
            .header = std.mem.bytesAsValue(Header, bytes[0..n]),
            .payload = bytes[n..],
        };

        if (!(msg.header.magic[0] == 'Z' and msg.header.magic[1] == 'M' and msg.header.magic[2] == 'L' and @intFromEnum(msg.header.model_id) < 4)) {
            return error.InvalidHeader;
        }

        return msg;
    }
};

const Request = struct {
    bytes: []u8,
    header: *align(1) Header,
    q: []u8,
    k: []u8,
    v: []u8,

    fn fromBytes(bytes: []u8, q_len: usize, k_len: usize, v_len: usize) !Request {
        const message = try Message.fromBytes(bytes);
        if (message.payload.len != q_len + k_len + v_len) return error.ShortMessage;

        return .{
            .bytes = message.bytes,
            .header = message.header,
            .q = message.payload[0..q_len],
            .k = message.payload[q_len .. q_len + k_len],
            .v = message.payload[q_len + k_len ..],
        };
    }
};

const Response = Message;

const Header = extern struct {
    const MAGIC: [3]u8 = .{ 'Z', 'M', 'L' }; // 0x5A4D4C

    const Type = enum(u8) {
        // Compute attention
        attn = 'a',
        // Starts a conversation
        start = 's',
        // Ping: the server answers immediately skipping all compute
        ping = 'p',
        _,
    };

    type: Type,
    kv_head_id: u8,

    conversation_id: u64 align(1),
    token_pos: u32 align(1),
    layer_id: u16 align(1),

    model_id: ModelId,
    first_q_id: u8,
    num_queries: u8,
    magic: [3]u8 align(1) = MAGIC,

    comptime {
        // destination MAC (6) + source MAC (6) + Ethertype (2) [+ 802.1Q / VLAN tag (4)]
        // We the optional VLAN, we can only guarantee an alignment of 4 for the payload after the prefix + header.
        const ETH_HEADER_SIZE = 6 + 6 + 2;
        const IP_HEADER_SIZE = 20;
        const UDP_HEADER_SIZE = 8;
        const PREFIX_SIZE = ETH_HEADER_SIZE + IP_HEADER_SIZE + UDP_HEADER_SIZE;
        stdx.debug.assertComptime(
            (PREFIX_SIZE + @sizeOf(Header)) % 4 == 0,
            "ETH_HEADER_SIZE ({}) + @sizeOf(AttnRequest) ({}) == {d} != k * 4",
            .{ PREFIX_SIZE, @sizeOf(Header), PREFIX_SIZE + @sizeOf(Header) },
        );
    }
};

const ModelId = enum(u8) {
    @"llama-3.1-8B" = 0,
    @"llama-3.2-1B" = 1,
    @"qwen3-14B" = 2,
    @"qwen3-32B" = 3,
    @"lfm2.5-1.2B" = 4,
};
