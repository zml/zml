# EXPERIMENTAL: Ascend NPU

This directory contains the Ascend NPU backend for ZML, loaded as a PJRT
plugin from the [XLA-NPU](https://github.com/Atlas-Ascend-Fork/xla-npu)
project.

# Getting Started

## Prerequisites

- Linux on x86_64 or aarch64 and an Ascend NPU visible under `/dev/davinci*`
- A matching CANN toolkit
- XLA-NPU built, producing `pjrt_c_api_npu_plugin.so`

Source the CANN environment and XLA-NPU's generated environment before running:

```bash
source /usr/local/Ascend/cann/set_env.sh
source /path/to/xla-npu/build/xla_npu_env
```

## Select the plugin

The loader uses `ZML_ASCEND_PJRT_PLUGIN` when set:

```bash
export ZML_ASCEND_PJRT_PLUGIN=/path/to/xla-npu/build/code/xla/bazel-bin/xla/xla_npu/pjrt/c/pjrt_c_api_npu_plugin.so
```

Otherwise it resolves the plugin from `XLA_NPU_ROOT_DIR`, which
`build/xla_npu_env` exports.

Set `SOC_VERSION` as required by XLA-NPU, for example:

```bash
export SOC_VERSION=$(python -c "import acl; acl.init(); print(acl.get_soc_name())")
```

## Run

```bash
bazel run //examples/ascend_smoke \
    --@zml//platforms:cpu=false \
    --@zml//platforms:ascend=true
```

# Status

Validated on 8x Ascend 950PR with CANN 9.1.0, XLA-NPU (openxla/xla
`a7e5887`-era master), and the `pjrt_c_api_npu_plugin.so` built from it.

Current limitations:

- Single device only. XLA-NPU does not implement Sharding/sdy ops yet, so ZML
  skips sharding constraints on this target.
- XLA-NPU rejects general reshapes (for example `u8[28,28] -> u8[784]`) in its
  AFIR path. Its ACLNN path now handles reshape and cast, but still lacks other
  ops (for example `iota`), so `//examples/mnist` does not run yet.
- XLA-NPU needs its NFusion pipeline to tolerate tuple-shaped producers and to
  simplify trivial tuple roots before the AFIR converter runs.

`//examples/ascend_smoke` is the smallest end-to-end check. On the setup above
it prints `ascend result: 2 6 12 20`.
