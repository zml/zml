#!/usr/bin/env bash

set -euo pipefail

latency_ms=${LATENCY_MS:-1000}
speed_mib=${SPEED_MIB:-100}
aws_endpoint=http://127.0.0.1:7878

if [[ ! "${latency_ms}" =~ ^[0-9]+$ ]] || [[ ! "${speed_mib}" =~ ^[0-9]+$ ]]; then
    echo "LATENCY_MS and SPEED_MIB must be non-negative integers" >&2
    exit 2
fi

speed_bytes_per_ms=$((speed_mib * 1024 * 1024 / 1000))

proxy_pid=

cleanup() {
    local status=$?

    trap - EXIT
    if [[ -n "${proxy_pid}" ]] && kill -0 "${proxy_pid}" 2>/dev/null; then
        kill "${proxy_pid}" 2>/dev/null || true
        wait "${proxy_pid}" 2>/dev/null || true
    fi

    exit "${status}"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

/nix/store/v8hmmywjqi94rzxzkqa0000j10ldsmcq-temurin-jre-bin-21.0.11/bin/java \
    -Ds3proxy.authorization='none' \
    "-Ds3proxy.endpoint=${aws_endpoint}" \
    "-Ds3proxy.latency-blobstore.*.latency=${latency_ms}" \
    "-Ds3proxy.latency-blobstore.*.speed=${speed_bytes_per_ms}" \
    -Djclouds.provider='filesystem' \
    -Djclouds.filesystem.basedir='/Users/brabier/s3proxy/data' \
    -jar /Users/brabier/s3proxy/s3proxy \
    --properties /dev/null &
proxy_pid=$!

# Do not race the benchmark against the proxy startup. Also fail promptly if the
# proxy exits before becoming ready.
for _ in {1..100}; do
    if ! kill -0 "${proxy_pid}" 2>/dev/null; then
        wait "${proxy_pid}"
        echo "s3proxy exited before becoming ready" >&2
        exit 1
    fi

    if curl --silent --fail --output /dev/null "${aws_endpoint}/"; then
        break
    fi

    sleep 0.1
done

if ! curl --silent --fail --output /dev/null "${aws_endpoint}/"; then
    echo "timed out waiting for s3proxy" >&2
    exit 1
fi

AWS_ENDPOINT_URL="${aws_endpoint}" ./bazel.sh run --config=release //examples/io:playground -- load s3://lfm
