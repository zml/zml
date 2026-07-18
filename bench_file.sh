#!/usr/bin/env bash

set -euo pipefail

CUDA_VISIBLE_DEVICES=1 ./bazel.sh run --config=release --@zml//platforms:cuda=true //examples/io:playground -- load ~/s3proxy/data/lfm/
