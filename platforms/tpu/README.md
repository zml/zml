bazel build //platforms/tpu:sandbox 
bazel run //platforms/tpu:requirements.update
bazel-bin/platforms/tpu/sandbox/bin/gen_ir_zig
