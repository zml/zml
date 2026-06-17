pub const inference = @import("step3_5flash/inference.zig");

pub const CompilationParameters = inference.CompilationParameters;
pub const CompilationOptions = inference.CompilationParameters;
pub const CompiledModel = inference.CompiledModel;

pub const model = @import("step3_5flash/model.zig");

pub const Config = model.Config;

pub const Buffers = model.Buffers;
pub const Model = model.Model;

pub const LoadedModel = model.LoadedModel;
// TODO: remove
pub const default_config = model.default_config;
