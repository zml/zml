const std = @import("std");

const platforms = @import("platforms");

const zml = @import("../../zml.zig");
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
pub fn attention(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor, parameters: Parameters) zml.Tensor {
    const q_sharded = q.withPartitioning(.{ .q = .replicated, .h = .model, .hd = .replicated });
    const k_sharded = k.withPartitioning(.{ .k = .replicated, .h = .model, .hd = .replicated });
    const v_sharded = v.withPartitioning(.{ .k = .replicated, .h = .model, .hd = .replicated });

    if (q.dim(.q) == 1) {
        const token_index_2d = token_index.broad(zml.Shape.init(.{ .row = 1, .col = 1 }, token_index.dtype()))
            .withPartitioning(.{ .row = .replicated, .col = .replicated });

        return zml.ops.manualComputation(
            .{ q_sharded, k_sharded, v_sharded, token_index_2d },
            q_sharded.shape(),
            parameters,
            (struct {
                fn body(context: Parameters, _: std.mem.Allocator, sharded_inputs: []const zml.Tensor, output: zml.Shape) zml.Tensor {
                    stdx.debug.assert(sharded_inputs.len == 4, "NKI decode attention expects 4 sharded inputs, got {}", .{sharded_inputs.len});
                    const kernel = context.decode_kernel;
                    return zml.ops.neuronNki(
                        .{ sharded_inputs[0], sharded_inputs[1], sharded_inputs[2], sharded_inputs[3] },
                        .{output},
                        .{
                            .name = kernel.name,
                            .entrypoint = kernel.entrypoint,
                            .source_path = kernel.source_path,
                            .compiler_target = context.compiler_target,
                        },
                    )[0];
                }
            }).body,
        ).convert(q.dtype());
    }

    return zml.ops.manualComputation(
        .{ q_sharded, k_sharded, v_sharded },
        q_sharded.shape(),
        parameters,
        (struct {
            fn body(context: Parameters, _: std.mem.Allocator, sharded_inputs: []const zml.Tensor, output: zml.Shape) zml.Tensor {
                const kernel = context.prefill_kernel;
                return zml.ops.neuronNki(
                    .{ sharded_inputs[0], sharded_inputs[1], sharded_inputs[2] },
                    .{output},
                    .{
                        .name = kernel.name,
                        .entrypoint = kernel.entrypoint,
                        .source_path = kernel.source_path,
                        .compiler_target = context.compiler_target,
                    },
                )[0];
            }
        }).body,
    ).convert(q.dtype());
}
