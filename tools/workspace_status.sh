#!/bin/bash
revision="$(git describe --always --dirty 2>/dev/null || echo dev)"
echo BUILD_SCM_REVISION "$revision"

echo STABLE_ZML_SMI_VERSION "$(git describe --tags --always --dirty --match 'zml-smi-v*' 2>/dev/null || echo dev)"
