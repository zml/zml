const std = @import("std");
const stdx = @import("stdx");

const zml = @import("../../zml.zig");
const bindings = @import("bindings.zig");

pub const load = bindings.load;
pub const register = bindings.register;

pub const fa2 = struct {
    pub const Parameters = struct {
        pub const InitOptions = struct {};

        pub fn init(opts: InitOptions) fa2.Parameters {
            _ = opts; // autofix
            return .{};
        }
    };

    pub const Metadata = struct {
        softmax_lse: zml.Tensor,
        softmax_lse_accum: zml.Tensor,
        out_accum: zml.Tensor,

        pub const InitOptions = struct {
            seqlen: i64,
        };

        pub fn init(opts: InitOptions) fa2.Metadata {
            return .{
                .softmax_lse = .init(.{1 * 32 * opts.seqlen * 4}, .i8),
                .softmax_lse_accum = .init(.{32 * 1 * 32 * 4 * 4}, .i8),
                .out_accum = .init(.{32 * 1 * 32 * 4 * 128 * 4}, .i8),
            };
        }

        pub fn initBuffer(self: fa2.Metadata, io: std.Io, platform: zml.Platform) !zml.Bufferized(fa2.Metadata) {
            return .{
                .softmax_lse = try zml.Buffer.uninitialized(io, platform, self.softmax_lse.shape(), .{}),
                .softmax_lse_accum = try zml.Buffer.uninitialized(io, platform, self.softmax_lse_accum.shape(), .{}),
                .out_accum = try zml.Buffer.uninitialized(io, platform, self.out_accum.shape(), .{}),
            };
        }

        pub fn deinitBuffer(self: *zml.Bufferized(fa2.Metadata)) void {
            self.softmax_lse.deinit();
            self.softmax_lse_accum.deinit();
            self.out_accum.deinit();
        }
    };

    pub fn attention(q_: zml.Tensor, k_: zml.Tensor, v_: zml.Tensor, token_index: zml.Tensor, metadata: fa2.Metadata, parameters: fa2.Parameters) zml.Tensor {
        _ = parameters; // autofix
        stdx.debug.assert(q_.shape().hasTag(.b) == null or q_.dim(.b) == 1, "fa2.attention support for batch size != 1 is not supported yet.", .{});
        const seqused_k = token_index.addConstant(q_.dim(.q)).reshape(.{1});
        // TODO(Corendos): replace with cumsum
        const cu_seqlens_k = b: {
            const zero = zml.Tensor.constant(token_index.dtype().zero()).reshape(.{1});
            break :b zml.Tensor.concatenate(&.{ zero, seqused_k }, 0).convert(.i32);
        };
        const max_seqlen_q: i32 = @intCast(q_.dim(.q));
        const max_seqlen_k: i32 = @intCast(k_.dim(.k));
        var q = q_.insertAxes(.q, .{.b}).merge(.{ .tot = .{ .b, .q } });
        const k = k_.insertAxes(.k, .{.b}).merge(.{ .tot = .{ .b, .k } });
        const v = v_.insertAxes(.k, .{.b}).merge(.{ .tot = .{ .b, .k } });
        // TODO(Corendos): replace with cumsum
        const cu_seqlens_q = zml.Tensor.constantTensor(zml.Shape.init(.{2}, .i32), std.mem.sliceAsBytes(&[2]i32{ 0, max_seqlen_q }));

        const original_tot = q.dim(.tot);
        const num_heads = q_.dim(.h);
        const num_heads_k = k_.dim(.h);
        const head_size = q_.dim(.hd);
        const ngroups = @divExact(num_heads, num_heads_k);
        const seqlenq_ngroups_swapped = max_seqlen_q == 1 and num_heads > num_heads_k and @mod(head_size, 8) == 0;
        if (seqlenq_ngroups_swapped) {
            q = q.splitAxis(.h, .{ .h = num_heads_k, .ngroups = ngroups }).transpose(.{ .tot, .ngroups, .h, .hd }).merge(.{ .tot = .{ .tot, .ngroups } });
        }

        var o = zml.ops.customCall(bindings.Fa2Varlen.custom_call_name, .{
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            metadata.softmax_lse,
            metadata.softmax_lse_accum,
            metadata.out_accum,
        }, .{q.shape()}, .{
            //.softmax_scale = @as(f32, 1.0),
            .is_causal = true,
            .window_size_left = @as(i32, -1),
            .window_size_right = @as(i32, -1),
            .max_seqlen_q = max_seqlen_q,
            .max_seqlen_k = max_seqlen_k,
            .num_heads = @as(i32, @intCast(num_heads)),
        }, .{
            .output_operand_aliases = &.{0},
            .has_side_effect = true,
        });

        if (seqlenq_ngroups_swapped) {
            o = o.splitAxis(.tot, .{ .tot = original_tot, .ngroups = ngroups }).transpose(.{ .tot, .h, .ngroups, .hd }).merge(.{ .h = .{ .h, .ngroups } });
        }

        return o.splitAxis(.tot, .{ .b = 1, .q = q_.dim(.q) }).squeeze(.b);
    }
};

pub const fa3 = struct {
    pub const Parameters = struct {
        pub const InitOptions = struct {};

        pub fn init(opts: InitOptions) fa3.Parameters {
            _ = opts; // autofix
            return .{};
        }
    };

    pub const Metadata = struct {
        softmax_lse: zml.Tensor,
        softmax_lse_accum: zml.Tensor,
        out_accum: zml.Tensor,
        scheduler_metadata: zml.Tensor,

        pub const InitOptions = struct {
            seqlen: i64,
        };

        pub fn init(opts: InitOptions) fa3.Metadata {
            return .{
                .softmax_lse = .init(.{1 * 32 * opts.seqlen * 4}, .i8),
                .softmax_lse_accum = .init(.{32 * 1 * 32 * 4 * 4}, .i8),
                .out_accum = .init(.{32 * 1 * 32 * 4 * 128 * 4}, .i8),
                .scheduler_metadata = .init(.{2}, .i32),
            };
        }

        pub fn initBuffer(self: fa3.Metadata, io: std.Io, platform: zml.Platform) !zml.Bufferized(fa3.Metadata) {
            return .{
                .softmax_lse = try zml.Buffer.uninitialized(io, platform, self.softmax_lse.shape(), .{}),
                .softmax_lse_accum = try zml.Buffer.uninitialized(io, platform, self.softmax_lse_accum.shape(), .{}),
                .out_accum = try zml.Buffer.uninitialized(io, platform, self.out_accum.shape(), .{}),
                .scheduler_metadata = try zml.Buffer.uninitialized(io, platform, self.scheduler_metadata.shape(), .{}),
            };
        }

        pub fn deinitBuffer(self: *zml.Bufferized(fa3.Metadata)) void {
            self.softmax_lse.deinit();
            self.softmax_lse_accum.deinit();
            self.out_accum.deinit();
            self.scheduler_metadata.deinit();
        }
    };

    pub fn attention(q_: zml.Tensor, k_: zml.Tensor, v_: zml.Tensor, token_index: zml.Tensor, metadata: fa3.Metadata, parameters: fa3.Parameters) zml.Tensor {
        _ = parameters; // autofix
        stdx.debug.assert(q_.shape().hasTag(.b) == null or q_.dim(.b) == 1, "fa3.attention support for batch size != 1 is not supported yet.", .{});
        const seqused_k = token_index.addConstant(q_.dim(.q)).reshape(.{1});
        // TODO(Corendos): replace with cumsum
        const cu_seqlens_k = b: {
            const zero = zml.Tensor.constant(token_index.dtype().zero()).reshape(.{1});
            break :b zml.Tensor.concatenate(&.{ zero, seqused_k }, 0).convert(.i32);
        };
        const max_seqlen_q: i32 = @intCast(q_.dim(.q));
        const max_seqlen_k: i32 = @intCast(k_.dim(.k));
        var q = q_.insertAxes(.q, .{.b}).merge(.{ .tot = .{ .b, .q } });
        const k = k_.insertAxes(.k, .{.b}).merge(.{ .tot = .{ .b, .k } });
        const v = v_.insertAxes(.k, .{.b}).merge(.{ .tot = .{ .b, .k } });
        // TODO(Corendos): replace with cumsum
        const cu_seqlens_q = zml.Tensor.constantTensor(zml.Shape.init(.{2}, .i32), std.mem.sliceAsBytes(&[2]i32{ 0, max_seqlen_q }));

        var o = zml.ops.customCall(bindings.Fa3.custom_call_name, .{
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            metadata.softmax_lse,
            metadata.softmax_lse_accum,
            metadata.out_accum,
            metadata.scheduler_metadata,
        }, .{q.shape()}, .{
            .is_causal = true,
            .window_size_left = @as(i32, -1),
            .window_size_right = @as(i32, -1),
            .max_seqlen_q = max_seqlen_q,
            .max_seqlen_k = max_seqlen_k,
        }, .{
            .output_operand_aliases = &.{0},
            .has_side_effect = true,
        });

        return o.splitAxis(.tot, .{ .b = 1, .q = q_.dim(.q) }).squeeze(.b);
    }
};
