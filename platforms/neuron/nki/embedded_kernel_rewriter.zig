const std = @import("std");

const c = @import("c");
const upb = @import("upb");

const custom_call_config = @import("custom_call_config.zig");
const custom_call_mutator = @import("custom_call_mutator.zig");
const kernel_compiler = @import("kernel_compiler.zig");

const log = std.log.scoped(.@"zml/platforms/neuron/nki/embedded_kernel_rewriter");

pub const zml_neuron_nki_target = "zml$neuron$nki";

// This file owns the embedded-kernel lowering boundary, not the whole-program
// Neuron compile flow.
//
// Input contract:
// - ZML emits synthetic `zml$neuron$nki` custom-calls in StableHLO.
// - Their backend_config carries the inline Python source, entrypoint, and
//   tensor signatures encoded in the ZML-specific text format parsed by
//   `custom_call_config.zig`.
//
// Output contract:
// - those synthetic calls are rewritten into `AwsNeuronCustomNativeKernel`
//   custom-calls that `neuronx-cc` understands
// - the rewritten backend_config payload is produced by `nki-cc` through a
//   single request/result JSON contract
// - frontend attributes are updated so outer parameter/root names match the
//   compiled NKI kernel IO names

// Rewrite every synthetic `zml$neuron$nki` custom-call in the module into
// `AwsNeuronCustomNativeKernel`. If none are present, return null so the caller
// can compile the original StableHLO unchanged.
pub fn rewriteCustomCalls(
    allocator: std.mem.Allocator,
    io: std.Io,
    tmp_dir: std.Io.Dir,
    hlo_code: []const u8,
    target: []const u8,
) !?[]const u8 {
    var upb_alloc: upb.Allocator = .init(allocator);
    const upb_arena = c.upb_Arena_Init(null, 0, upb_alloc.inner());

    const hlo_module = try upb.parse(c.xla_HloModuleProto, upb_arena, hlo_code);

    var rewritten: usize = 0;
    var kernel_index: usize = 0;

    var computations_len: usize = undefined;
    const computations = c.xla_HloModuleProto_mutable_computations(hlo_module, &computations_len)[0..computations_len];
    for (computations) |comp| {
        var instructions_len: usize = undefined;
        const instructions = c.xla_HloComputationProto_mutable_instructions(comp, &instructions_len)[0..instructions_len];
        for (instructions) |instruction| {
            const opcode = upb.slice(c.xla_HloInstructionProto_opcode(instruction)) orelse continue;
            if (!std.mem.eql(u8, opcode, "custom-call")) continue;

            const call_target = upb.slice(c.xla_HloInstructionProto_custom_call_target(instruction)) orelse continue;
            if (!std.mem.eql(u8, call_target, zml_neuron_nki_target)) continue;

            const encoded_backend_config = upb.slice(c.xla_HloInstructionProto_backend_config(instruction)) orelse return error.InvalidNkiBackendConfig;
            const kernel_config = try custom_call_config.parseKernelConfig(allocator, encoded_backend_config);
            log.info("Rewriting custom-call {s} as AwsNeuronCustomNativeKernel", .{kernel_config.name});
            const compiled_kernel = try kernel_compiler.compileKernel(allocator, io, tmp_dir, kernel_config, target, kernel_index);
            custom_call_mutator.rewriteInstructionAsCustomNativeKernel(instruction, compiled_kernel.backend_config_bytes);
            custom_call_mutator.setOuterCustomNativeKernelIoAttributes(
                allocator,
                upb_arena,
                comp,
                instructions,
                instruction,
                compiled_kernel.input_names,
                compiled_kernel.output_names,
            );

            rewritten += 1;
            kernel_index += 1;
        }
    }

    if (rewritten == 0) return null;
    log.info("Rewrote {d} ZML NKI custom-call(s) for target {s}", .{ rewritten, target });
    return try upb.serialize(hlo_module, upb_arena);
}
