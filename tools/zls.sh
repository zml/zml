#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"
cd "$(bazel info workspace)"
exec bazel run -- @zml//:zls "${@}"
