#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
cd "$(bazel info workspace)"
exec bazel run -- @zml//:completion "${@}"
