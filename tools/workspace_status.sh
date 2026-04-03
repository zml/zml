#!/bin/bash
if git diff --quiet; then
    echo BUILD_SCM_REVISION "$(git rev-parse --short HEAD)"
else
    echo BUILD_SCM_REVISION "dev"
fi

echo STABLE_ZML_SMI_VERSION "$(git describe --tags --dirty --match 'zml-smi-v*' 2>/dev/null || echo "dev")"
