bazel build --config=sycl --repo_env=SYCL_BUILD_HERMETIC=1 --repo_env=ONEAPI_VERSION=2026.0 --repo_env=OS=ubuntu_24.04 //cc/tests/gpu/sycl:vector_sycl_test
# bazel build --config=sycl_hermetic --config=icpx_clang @oneapi//:compiler_incs @oneapi//:libs @oneapi//:binaries @oneapi//:headers
# bazel build --config=sycl_hermetic --config=icpx_clang //cc/impls/linux_x86_64_linux_x86_64_sycl:all
# bazel build --config=sycl_hermetic --config=icpx_clang @local_config_sycl//sycl:build_defs_bzl

bazel build //xla/pjrt/c:pjrt_c_api_gpu_plugin.so --config=sycl_hermetic

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

# bazel build \
#   --config=sycl \
#   --repo_env=SYCL_BUILD_HERMETIC=1 \
#   --repo_env=ONEAPI_VERSION=2026.0 \
#   --repo_env=OS=ubuntu_24.04 \
#   --copt=-Wno-unknown-warning-option \
#   --host_copt=-Wno-unknown-warning-option //xla/backends/gpu/collectives:oneccl_collectives


cd /home/kevin/xla

  --override_repository=rules_ml_toolchain=/home/kevin/rules_ml_toolchain \
bazel build \
  --config=sycl_hermetic \
  --config=icpx_clang \
  //xla/pjrt/c:pjrt_oneapi_plugin_sandbox

chmod u+w /home/kevin/zml/xla/libpjrt_oneapi/libpjrt_oneapi.so &&
cp /home/kevin/xla/bazel-bin/xla/pjrt/c/pjrt_c_api_gpu_plugin.so /home/kevin/zml/xla/libpjrt_oneapi/libpjrt_oneapi.so

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



cd /home/kevin/zml
mkdir -p /tmp/zml-unitrace/always

XLA_FLAGS='--xla_gpu_command_buffer_update_mode=ALWAYS_UPDATE' \
bazel run \
  --run_under='/home/kevin/pti-gpu/tools/unitrace/build/unitrace --chrome-kernel-logging --chrome-sycl-logging --chrome-ccl-logging --chrome-call-logging --kernel-submission --device-timeline --ccl-summary-report --output-dir-path /tmp/zml-unitrace/always --output always' \
  --override_repository=+oneapi_packages+libpjrt_oneapi=/home/kevin/zml/xla/libpjrt_oneapi \
  --config=release \
  --@zml//platforms:cpu=false \
  --@zml//platforms:oneapi=true \
  //examples/llm -- \
  --model=/var/models/meta-llama/Llama-3.1-8B-Instruct \
  --topk=4 \
  --prompt='Tell me a story about cats.'


ONEAPI_DEVICE_SELECTOR=level_zero:0 \
bazel run //examples/llm \
  --config=release \
  --config=sycl_hermetic \
  --@zml//platforms:cpu=false \
  --@zml//platforms:oneapi=true \
  --override_repository=+oneapi_packages+libpjrt_oneapi=/home/kevin/zml/xla/libpjrt_oneapi \
  --override_repository=+non_module_deps+xla=/home/kevin/xla \
  -- \
  --model=/var/models/meta-llama/Llama-3.1-8B-Instruct \
  --topk=2 \
  --prompt="Tell me a story in about a cat"
