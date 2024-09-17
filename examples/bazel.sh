#!/bin/bash
BAZELISK_VERSION=v1.20.0

case $OSTYPE in
    "darwin"*)
        OS="darwin"
        CACHE_DIR="${HOME}/Library/Caches/bazelisk"
        ;;
    "linux"*)
        OS="linux"
        if [[ -n "${XDG_CACHE_HOME}" ]]; then
            CACHE_DIR="${XDG_CACHE_HOME}/bazelisk"
        else
            CACHE_DIR="${HOME}/.cache/bazelisk"
        fi
        ;;
esac

case $(uname -m) in
    "arm64")
        ARCH="arm64"
        ;;
    "x86_64")
        ARCH="amd64"
        ;;
esac

BAZELISK="${CACHE_DIR}/bazelisk-${BAZELISK_VERSION}"

if [[ ! -f "${BAZELISK}" ]]; then
    mkdir -p "${CACHE_DIR}"
    curl -L -o "${CACHE_DIR}/bazelisk-${BAZELISK_VERSION}" "https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-${OS}-${ARCH}"
    chmod +x "${BAZELISK}"
fi

exec "${BAZELISK}" "$@"
