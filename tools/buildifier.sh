#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"
exec bazel run -- @buildifier_prebuilt//:buildifier "$@"
