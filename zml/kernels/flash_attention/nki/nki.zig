const platforms = @import("platforms");

const zml = @import("zml/core");
const stdx = @import("stdx");

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
                        .source_path = "zml/zml/kernels/flash_attention/nki/attention.py",
                    },
                },
                .prefill_kernel = switch (compiler_target) {
                    .trn1, .inf2, .trn1n, .trn2, .trn2n, .trn3 => .{
                        .name = "nki_attention_trn1_prefill",
                        .entrypoint = "prefill",
                        .source_path = "zml/zml/kernels/flash_attention/nki/attention.py",
                    },
                },
            };
        }

        stdx.debug.panic("NKI attention requires the Neuron platform", .{});
    }
};

/// Attention for Neuron. Both prefill and decode lower to NKI custom calls.
pub fn attention(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor, parameters: Parameters) zml.Tensor {
    const decode = q.dim(.q) == 1;
    const kernel = if (decode) parameters.decode_kernel else parameters.prefill_kernel;
    const options: zml.ops.NeuronNkiOps = .{
        .name = kernel.name,
        .entrypoint = kernel.entrypoint,
        .source_path = kernel.source_path,
        .compiler_target = parameters.compiler_target,
    };
    const output = if (decode)
        zml.ops.neuronNki(.{ q, k, v, token_index.broad(zml.Shape.init(.{ .row = 1, .col = 1 }, token_index.dtype())) }, .{q.shape()}, options)[0]
    else
        zml.ops.neuronNki(.{ q, k, v }, .{q.shape()}, options)[0];
    return output.convert(q.dtype());
}
