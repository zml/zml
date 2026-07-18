#!/usr/bin/env bash

set -euo pipefail

./bazel.sh run --config=release //examples/io:playground -- load ~/s3proxy/data/lfm/
