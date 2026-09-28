# Furiosa typed TCL backend

The sole `furiosa` platform uses the typed TCL backend (formerly `furiosa3`).
The default is four PEs; use `--furiosa-pe-count=8` for eight-PE Llama runs.
Select topology before creating the first client, in a separate process for each topology.
Tensor placement annotations, pinned-host transfers, plugin-provided layouts,
Shardy, asynchronous execution, vanilla attention and StableHLO paged attention
are retained. Triton attention is unavailable on this platform.

## Build and package the PJRT plugin

Use each checkout's pinned Bazel: XLA 8.7.0 and ZML 9.1.1.

```sh
cd /home/kevin/furiosa/xla-private
bazel build --config=hermetic_linux_x86 --jobs=16 \
  //xla/pjrt/furiosa:pjrt_c_api_furiosa_plugin

mkdir -p /home/kevin/furiosa/xla-override/lib
ln -sfn /home/kevin/furiosa/xla-private/bazel-bin/xla/pjrt/furiosa/libpjrt_c_api_furiosa_plugin.so \
  /home/kevin/furiosa/xla-override/lib/libpjrt_c_api_furiosa_plugin.so
```

The override needs empty `MODULE.bazel` and `REPO.bazel` files and this `BUILD.bazel`:

```starlark
filegroup(
    name = "libzml_furiosa",
    srcs = ["lib/libpjrt_c_api_furiosa_plugin.so"],
    visibility = ["//visibility:public"],
)
```

The repository override supplies the already-built library as runtime data.
It does not build XLA or install an SDK. The loader follows the plugin symlink
before loading so XLA's `$ORIGIN` paths still find its toolchain libraries
(including `libunwind.so.1`). Keep the XLA build output and its `_solib` directory
available; copying only the `.so` is not a standalone distribution. Rebuild XLA before running ZML after
plugin changes; the symlink exposes the new output without a separate copy.
No absolute local path is stored in ZML's module configuration.
CPU-only builds need neither this override nor the Furiosa SDK.
Enabling Furiosa without an override reports a missing-plugin diagnostic.

## Run Llama

The inspected plugin requires TCC 2026.3.0 and runtime
`xla-furiosa-opt-rt/0.8.1 bridge/19`, executable format 7, topology generation 1.
The temporary SDK below must exist, or be replaced with a compatible installation.
The runtime/compiler are separate from the PJRT shared library.

```sh
cd /home/kevin/furiosa/zml
export XLA_FURIOSA_COMPILER=/tmp/furiosa-integrated-sdk/furiosa-tcc
export XLA_FURIOSA_RUNTIME_LIBRARY=/tmp/furiosa-integrated-sdk/libdevice_runtime.so
export XLA_FURIOSA_COMPILER_CACHE=/home/kevin/.local/state/xla-rngd/furiosa/tcl-ir-v7/zml
export XLA_FURIOSA_VISIBLE_DEVICES=0
unset XLA_FURIOSA_PJRT_LIBRARY

bazel run --override_repository=libzml_furiosa=/home/kevin/furiosa/xla-override \
  --jobs=16 --@zml//platforms:furiosa=true --@zml//platforms:cpu=false \
  //examples/llm -- \
  --model=/var/models/meta-llama/Llama-3.1-8B-Instruct \
  --furiosa-pe-count=8 --backend=vanilla --seqlen=128 --topk=1 \
  --prompt='Count from 1 to 20, separated by commas.'
```

The loader resolves `libzml_furiosa/lib/libpjrt_c_api_furiosa_plugin.so` using
Bazel runfiles and repository mapping, independent of the current directory.
`XLA_FURIOSA_PJRT_LIBRARY` optionally takes precedence for explicit debugging;
an invalid path is an error, not a fallback. It does not bypass Bazel packaging.
Retired variant flags, enums and environment-variable aliases are removed.

Use a fresh compiler-cache namespace and recompile old serialized executables.
Multi-chip runs require eight PEs and ascending visible devices, e.g. `0,1`.
Pinned-host transfers require access to `/dev/dma_heap/system` and RNGD devices.
Task linking and pooling retain plugin defaults; optional fusion remains opt-in.
The XLA `run_llama.sh` environment-based launcher is outside this migration.

## Validation

See [VALIDATION.md](VALIDATION.md) for results and limitations of this migration.
For CPU references, enable CPU alongside Furiosa and use
`//examples/llm:llama_tests -- --platform=furiosa --furiosa-pe-count=4
--compare-cpu --forward-only --layers=32 --seqlen=1 --cache-seqlen=128`
with the same override and model argument. Repeat with eight PEs in a new process.
The generation executable does not accept `--platform`.
Bazel device tests need explicit `--test_env` forwarding for compiler, runtime,
cache and visible-device variables. Generic tests use the four-PE default.
Historical throughput figures do not establish current correctness or performance.

For focused eight-PE pinned transfers, repeated execution and input ownership, run this separate test process with the SDK exports above:

```sh
bazel test --override_repository=libzml_furiosa=/home/kevin/furiosa/xla-override \
  --jobs=16 --config=debug --@zml//platforms:furiosa=true --@zml//platforms:cpu=false \
  --test_env=XLA_FURIOSA_COMPILER --test_env=XLA_FURIOSA_RUNTIME_LIBRARY \
  --test_env=XLA_FURIOSA_COMPILER_CACHE --test_env=XLA_FURIOSA_VISIBLE_DEVICES=0 \
  //zml:furiosa_8pe_test
```

Repeat with `--test_env=XLA_FURIOSA_VISIBLE_DEVICES=0,1` for the small two-chip
sharding gate. This manual target fixes PE count at eight and does not alter
production defaults or the generic test suite's four-PE topology.
