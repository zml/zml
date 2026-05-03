#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
cd "$(bazel info workspace)"

# Only provided in Bazel 9
export BUILD_EXECROOT="$(bazel info execution_root)"

exec bazel run -- @zml//:completion "${@}"
