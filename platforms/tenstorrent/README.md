# Tenstorrent

Enable the experimental Tenstorrent PJRT backend with
`--@zml//platforms:tenstorrent=true`. The initial integration loads a locally
built plugin; it does not download a prebuilt SDK.

Build these targets in the XLA checkout on branch `steeve/tt`:

```sh
bazel build --config=hermetic_linux_x86 \
  //xla/pjrt/tenstorrent:pjrt_c_api_tenstorrent_plugin \
  //xla/pjrt/tenstorrent:hardware_test
```

Set `ZML_TENSTORRENT_PLUGIN_PATH` to the absolute path of
`bazel-bin/xla/pjrt/tenstorrent/libpjrt_c_api_tenstorrent_plugin.so` and
`TT_METAL_RUNTIME_ROOT` to the `+tenstorrent+tt_metal` directory inside
`bazel-bin/xla/pjrt/tenstorrent/hardware_test.runfiles`. That runfiles directory
combines device sources, SFPI, and generated firmware support objects. Preserve
the plugin's relative library layout for its libnuma and libdwarf dependencies.

The host needs Linux x86-64, the Tenstorrent driver, and configured hugepages.
The backend currently compiles single-device programs. ZML selects the first
addressable device by default; `CreateOptions.tenstorrent.device_index` selects
another device. Disable the CPU platform so a failed accelerator initialization
cannot fall back to CPU:

```sh
bazel run \
  --@zml//platforms:tenstorrent=true --@zml//platforms:cpu=false \
  //examples/mnist

bazel run \
  --@zml//platforms:tenstorrent=true --@zml//platforms:cpu=false \
  //examples/llm -- --model=/path/to/llama --prompt="Hello" \
  --backend=stablehlo --seqlen=64 --topk=1
```

Host transfers use buffered copies. Pinned host memory, external device pointers,
and GPU attention kernels are not used by this integration. Model execution is
subject to the operation and dtype coverage of the experimental XLA backend.
