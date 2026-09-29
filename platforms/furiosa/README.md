# Furiosa typed TCL backend

The sole `furiosa` platform uses the typed TCL backend (formerly `furiosa3`).
The default is four PEs; use `--furiosa-pe-count=8` for eight-PE Llama runs.
Select topology before creating the first client, in a separate process for each
configuration. Multi-chip execution requires eight PEs. Triton attention is
unavailable; vanilla and StableHLO paged attention remain supported.

## Build and stage the project libraries

Use each checkout's pinned Bazel: XLA 8.7.0 and ZML 9.1.1.

```sh
cd /home/kevin/furiosa/xla-private
bazel build --config=hermetic_linux_x86 --jobs=16 \
  //xla/pjrt/furiosa:pjrt_c_api_furiosa_plugin
```

This builds the PJRT plugin, not its custom Rust runtime bridge. Build the bridge
when first preparing the override, or when its contract/generated kernels change:

```sh
cd /home/kevin/furiosa/xla-private
bash xla/pjrt/furiosa/install_compiler.sh /home/kevin/furiosa/furiosa-sdk
bash xla/stream_executor/furiosa/install_sdk.sh /home/kevin/furiosa/furiosa-sdk
```

The existing installer needs host Rust/Cargo and AArch64 GCC. It builds
`xla-furiosa-opt-rt/0.8.1 bridge/19`, including copy/zero kernels. This preparation
step is separate from the application sandbox; ZML never invokes Cargo or these
installers. A matching already-built bridge can be used instead.

Stage real files into the override. Repeat after rebuilding the plugin or bridge:

```sh
cd /home/kevin/furiosa/xla-private
furiosa_plugin="$PWD/bazel-bin/xla/pjrt/furiosa/libpjrt_c_api_furiosa_plugin.so"
furiosa_override=/home/kevin/furiosa/xla-override
furiosa_sdk=/home/kevin/furiosa/furiosa-sdk
mkdir -p "$furiosa_override/lib"
touch "$furiosa_override/MODULE.bazel" "$furiosa_override/REPO.bazel"

install -m 0755 "$furiosa_plugin" "$furiosa_override/lib/libpjrt_c_api_furiosa_plugin.so.new"
mv -Tf "$furiosa_override/lib/libpjrt_c_api_furiosa_plugin.so.new" "$furiosa_override/lib/libpjrt_c_api_furiosa_plugin.so"
install -m 0755 "$furiosa_sdk/libdevice_runtime.so" "$furiosa_override/lib/libdevice_runtime.so.new"
mv -Tf "$furiosa_override/lib/libdevice_runtime.so.new" "$furiosa_override/lib/libdevice_runtime.so"
for furiosa_lib in libstdc++.so.6 libunwind.so.1; do
  furiosa_source=$(ldd "$furiosa_plugin" | awk -v name="$furiosa_lib" '$1 == name { print $3; exit }')
  test -f "$furiosa_source" || exit 1
  install -m 0755 "$furiosa_source" "$furiosa_override/lib/$furiosa_lib.new"
  mv -Tf "$furiosa_override/lib/$furiosa_lib.new" "$furiosa_override/lib/$furiosa_lib"
done
```

Create this `/home/kevin/furiosa/xla-override/BUILD.bazel`:

```starlark
exports_files([
    "lib/libpjrt_c_api_furiosa_plugin.so",
    "lib/libdevice_runtime.so",
    "lib/libstdc++.so.6",
    "lib/libunwind.so.1",
])

filegroup(
    name = "libzml_furiosa",
    srcs = glob(["lib/*.so*"]),
    visibility = ["//visibility:public"],
)
```

The plugin's C++ and unwind libraries come from its matching XLA build. The
runtime's `libgcc_s.so.1` comes from the pinned packages below. ZML patches copies
of the libraries, so the assembled bundle needs neither XLA's `_solib` directory
nor the original SDK directory. Staging replaces the previous symlink-only
workflow; a rebuilt XLA plugin is not picked up until its staged copy is refreshed.

## Automatic sandbox

Bazel downloads the official TCC 2026.3.0 wheel, verifies its checksum, and
extracts the native executable. No pip or Python environment is installed.
AArch64 GCC 13, binutils, headers, and required libraries are extracted from
checksum-locked Ubuntu Noble packages. Regenerate that lock with:

```sh
cd /home/kevin/furiosa/zml
bazel run @apt_furiosa//:lock
```

`//platforms/furiosa:sandbox` combines these dependencies with the override.
One compiled Zig launcher supplies both compiler entry points, `furiosa-tcc`
and `aarch64-linux-gnu-gcc`. It uses packaged tools, libraries, and a packaged
sysroot in the compiler process; no runtime shell launcher is needed.
The application uses runfiles to locate the bundle, including
when launched outside the checkout. The validated host baseline is Ubuntu 24.04
x86-64 (glibc 2.39). The host still provides glibc and its ELF loader,
the NPU driver, device permissions, and model files. CPU-only builds need no
Furiosa override. Enabling Furiosa without one reports a setup error.

The loader automatically sets the compiler and runtime paths, replacing inherited
`XLA_FURIOSA_COMPILER` and `XLA_FURIOSA_RUNTIME_LIBRARY` values. It always loads
the packaged plugin; `XLA_FURIOSA_PJRT_LIBRARY` is ignored.

The default visible chip is `0`. An explicit `XLA_FURIOSA_VISIBLE_DEVICES` remains
supported, including ascending `0,1` for multi-chip execution or an empty value
to hide all chips. An explicit `XLA_FURIOSA_COMPILER_CACHE` is also preserved.
Otherwise the cache is created under `TEST_TMPDIR`, `XDG_CACHE_HOME`, or
`$HOME/.cache`, in that order; without those variables an owned temporary
directory is used. The cache namespace is
`zml/furiosa/tcc-2026.3.0-bridge19-ir7-gcc13-v1`; bump it when changing the pinned
toolchain or runtime/image contract. The sandbox itself remains read-only.

## Run Llama

No Furiosa exports or plugin-override unset are required:

```sh
cd /home/kevin/furiosa/zml
bazel run --override_repository=libzml_furiosa=/home/kevin/furiosa/xla-override \
  --jobs=16 --config=debug \
  --@zml//platforms:furiosa=true --@zml//platforms:cpu=false \
  //examples/llm -- \
  --model=/var/models/meta-llama/Llama-3.1-8B-Instruct \
  --furiosa-pe-count=8 --backend=vanilla --seqlen=128 --topk=1 \
  --prompt='Count from 1 to 20, separated by commas.'
```

Old serialized executables require regeneration when their contract changes;
current executable format is 7 and topology generation is 1. Task linking and
pooling retain plugin defaults; optional fusion remains opt-in.

## Validation

Run the existing eight-PE hardware tests with the packaged compiler and runtime:

```sh
cd /home/kevin/furiosa/zml
bazel test --override_repository=libzml_furiosa=/home/kevin/furiosa/xla-override \
  --jobs=16 --config=debug \
  --@zml//platforms:furiosa=true --@zml//platforms:cpu=false \
  //zml:furiosa_8pe_test
```

No compiler/runtime/cache `--test_env` forwarding is needed. Repeat the hardware
test with `--test_env=XLA_FURIOSA_VISIBLE_DEVICES=0,1` for two-chip execution.
Pinned-host transfers require permission to access `/dev/dma_heap/system` and
the RNGD devices.
