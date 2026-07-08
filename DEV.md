HRX branch: zml/zml/hrx-system/rocm-jaxlib-v0.10.2-hrx
XLA branch: zml/xla/hugomano/rocm-jaxlib-v0.10.2-hrx


cd /home/hugo/hrx-system
bazel build -c opt --//runtime/config/hal:drivers=amdgpu,local-sync,local-task,null //libhrx/src/libhrx:hrx //libhrx/src/binding/hip:amdhip64

cd /home/hugo/xla
bazel build --spawn_strategy=local --disk_cache=/home/hugo/.cache/bazel-disk --config=rocm_ci --config=baseline_x86_64 //xla/pjrt/c:pjrt_c_api_gpu_plugin

Repack with 
```sh
cd /home/hugo/zml

HRX_ROOT=/home/hugo/hrx-system
XLA_ROOT=/home/hugo/xla

PJRT_SO="$(find "$XLA_ROOT/bazel-bin/xla/pjrt/c" -maxdepth 1 -type f -name 'libpjrt_c_api_gpu_plugin.so' -print -quit)"

rm -rf .local_pjrt_rocm
mkdir -p .local_pjrt_rocm/pkg/hrx

cp "$PJRT_SO" .local_pjrt_rocm/pkg/libpjrt_rocm.so
cp "$HRX_ROOT/bazel-bin/libhrx/src/libhrx/libhrx.so" .local_pjrt_rocm/pkg/hrx/libhrx.so.0
cp "$HRX_ROOT/bazel-bin/libhrx/src/binding/hip/libamdhip64.so" .local_pjrt_rocm/pkg/hrx/libamdhip64.so.7

tar --sort=name \
  --mtime='UTC 2026-01-01' \
  --owner=0 --group=0 --numeric-owner \
  -czf .local_pjrt_rocm/pjrt-rocm_linux-amd64.tar.gz \
  -C .local_pjrt_rocm/pkg .

sha256sum .local_pjrt_rocm/pjrt-rocm_linux-amd64.tar.gz
tar -tzf .local_pjrt_rocm/pjrt-rocm_linux-amd64.tar.gz
```

Copy the new hash into `platforms/rocm/rocm.bzl`:

```python
http_archive(
    name = "libpjrt_rocm_hrx",
    build_file = "libpjrt_rocm_hrx.BUILD.bazel",
    url = "file:///home/hugo/zml/.local_pjrt_rocm/pjrt-rocm_linux-amd64.tar.gz",
    sha256 = "<new sha256>",
)
```

Test with : 

cd /home/hugo/zml
HRX_MEM_POOL_BYTES=29077431808 bazel run //examples/benchmark --@zml//platforms:rocm_hrx=true
