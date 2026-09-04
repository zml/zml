#!/usr/bin/env bash
cd "$(dirname "${BASH_SOURCE[0]}")"
cd "$(bazel info workspace)"
exec bazel run -- @buildifier_prebuilt//:buildifier "$@"
