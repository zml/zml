bazel build --config=sycl --repo_env=SYCL_BUILD_HERMETIC=1 --repo_env=ONEAPI_VERSION=2026.0 --repo_env=OS=ubuntu_24.04 //cc/tests/gpu/sycl:vector_sycl_test
# bazel build --config=sycl_hermetic --config=icpx_clang @oneapi//:compiler_incs @oneapi//:libs @oneapi//:binaries @oneapi//:headers
# bazel build --config=sycl_hermetic --config=icpx_clang //cc/impls/linux_x86_64_linux_x86_64_sycl:all
# bazel build --config=sycl_hermetic --config=icpx_clang @local_config_sycl//sycl:build_defs_bzl

# archive=/home/kevin/rules-ml-toolchain-redists/dist/musa-toolkit-rc3.1.1-musa_sdk_rc3_1_1-ubuntu-x86_64.tar.zst
# sha=$(sha256sum "$archive" | awk '{print $1}')
# MUSA_DISTRO_URL="file://${archive}" \
# MUSA_DISTRO_HASH="$sha" \
# MUSA_DISTRO_ROOT=musa \
# MUSA_DEVICE=S80 \
# TF_NEED_MUSA=1 \
# bazel build //cc/tests/gpu/musa:vector_musa_build_test --config=musa


# cd /home/kevin/xla
# MUSA_VERSION=rc3.1.1 MUSA_GPU_ARCHS=mp_21 \
# bazel build --config=musa -c opt //xla/pjrt/c:pjrt_c_api_gpu_plugin.so


# cd /home/kevin/zml
# bazel build //zml \
#   --@zml//platforms:cpu=false \
#   --@zml//platforms:musa=true \
#   --override_repository=+musa_packages+libpjrt_musa=/home/kevin/zml/xla/

cd /home/kevin/xla

bazel build \
  --override_repository=rules_ml_toolchain=/home/kevin/rules_ml_toolchain \
  --config=sycl_hermetic \
  --config=icpx_clang \
  //xla/pjrt/c:pjrt_oneapi_plugin_sandbox


cd /home/kevin/zml


ONEAPI_DEVICE_SELECTOR=level_zero:0 \
bazel run \
  --override_repository=+oneapi_packages+libpjrt_oneapi=/home/kevin/zml/xla/libpjrt_oneapi \
  --config=release \
  --@zml//platforms:cpu=false \
  --@zml//platforms:oneapi=true \
  //examples/llm -- \
  --model=/var/models/meta-llama/Llama-3.1-8B-Instruct \
  --topk=4 \
  --prompt="What is the capital of France?"
