const std = @import("std");
const zml = @import("zml");
const CompileArgs = @import("../../compiler.zig").CompileArgs;

const Qwen3VLTextModel = @import("./Qwen3VLTextModel.zig");

const Qwen3VLModel = @This();

language_model: Qwen3VLTextModel,

pub fn unloadBuffers(self: *zml.Bufferized(Qwen3VLModel)) void {
    Qwen3VLTextModel.unloadBuffers(&self.language_model);
}

pub fn forward(self: Qwen3VLModel, input_ids: zml.Tensor, args: CompileArgs) zml.Tensor {
    return self.language_model.forward(input_ids, args);
}
