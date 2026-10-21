const platforms = @import("platforms");

const zml = @import("../../../zml.zig");
const stdx = zml.stdx;

const KernelSpec = struct {
    name: []const u8,
    entrypoint: []const u8,
    source_path: []const u8,
};

pub const Parameters = struct {
    compiler_target: []const u8,
    decode_kernel: KernelSpec,
    prefill_kernel: KernelSpec,

    pub fn init() Parameters {
        if (comptime platforms.isEnabled(.neuron)) {
            const nki_kernel = @import("platforms/neuron/nki_kernel");
            const compiler_target = nki_kernel.compilerTargetFromInstance();
            return .{
                .compiler_target = @tagName(compiler_target),
                .decode_kernel = switch (compiler_target) {
                    .trn1, .inf2, .trn1n, .trn2, .trn2n, .trn3 => .{
                        .name = "nki_attention_trn1_decode",
                        .entrypoint = "decode",
                        .source_path = "zml/zml/attention/nki/attention.py",
                    },
                },
                .prefill_kernel = switch (compiler_target) {
                    .trn1, .inf2, .trn1n, .trn2, .trn2n, .trn3 => .{
                        .name = "nki_attention_trn1_prefill",
                        .entrypoint = "prefill",
                        .source_path = "zml/zml/attention/nki/attention.py",
                    },
                },
            };
        }

        stdx.debug.panic("NKI attention requires the Neuron platform", .{});
    }
};

/// Attention for Neuron. Both prefill and decode lower to NKI custom calls.
pub fn attention(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor, parameters: Parameters, sharding: zml.attention.Sharding) zml.Tensor {
    stdx.debug.assert(sharding.heads == sharding.kv_heads, "This attention kernel requires matching query and KV head sharding", .{});
    if (q.dim(.q) == 1) {
        const token_index_2d = token_index.broad(zml.Shape.init(.{ .row = 1, .col = 1 }, token_index.dtype()));

        return zml.ops.manualComputation(
            (struct {
                q: zml.Tensor,
                k: zml.Tensor,
                v: zml.Tensor,
                token_index: zml.Tensor,
                parameters: Parameters,

                fn body(self: @This(), output: zml.Shape) zml.Tensor {
                    const kernel = self.parameters.decode_kernel;
                    return zml.ops.neuronNki(
                        .{ self.q, self.k, self.v, self.token_index },
                        .{output},
                        .{
                            .name = kernel.name,
                            .entrypoint = kernel.entrypoint,
                            .source_path = kernel.source_path,
                            .compiler_target = self.parameters.compiler_target,
                        },
                    )[0];
                }
            }).body,
            .{
                .q = q,
                .k = k,
                .v = v,
                .token_index = token_index_2d,
                .parameters = parameters,
            },
            q.shape(),
            .{ .mesh = sharding.mesh, .in_specs = .{ .q = sharding.partition(q.shape(), .h), .k = sharding.kvPartition(k.shape(), .h), .v = sharding.kvPartition(v.shape(), .h), .token_index = .replicated }, .out_specs = sharding.partition(q.shape(), .h) },
        ).convert(q.dtype());
    }

    return zml.ops.manualComputation(
        (struct {
            q: zml.Tensor,
            k: zml.Tensor,
            v: zml.Tensor,
            parameters: Parameters,

            fn body(self: @This(), output: zml.Shape) zml.Tensor {
                const kernel = self.parameters.prefill_kernel;
                return zml.ops.neuronNki(
                    .{ self.q, self.k, self.v },
                    .{output},
                    .{
                        .name = kernel.name,
                        .entrypoint = kernel.entrypoint,
                        .source_path = kernel.source_path,
                        .compiler_target = self.parameters.compiler_target,
                    },
                )[0];
            }
        }).body,
        .{ .q = q, .k = k, .v = v, .parameters = parameters },
        q.shape(),
        .{ .mesh = sharding.mesh, .in_specs = .{ .q = sharding.partition(q.shape(), .h), .k = sharding.kvPartition(k.shape(), .h), .v = sharding.kvPartition(v.shape(), .h) }, .out_specs = sharding.partition(q.shape(), .h) },
    ).convert(q.dtype());
}
