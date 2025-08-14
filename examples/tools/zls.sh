#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec bazel run --experimental_convenience_symlinks=ignore -- @zml//:zls "${@}"
