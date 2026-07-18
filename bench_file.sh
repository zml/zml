#!/usr/bin/env bash

set -euo pipefail

ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3 ./bazel.sh run --config=release --@zml//platforms:oneapi=true //examples/io:playground -- load ~/s3proxy/data/lfm/
