#!/bin/bash
set -euo pipefail

SCRIPT_PATTERNS=("*.sh" "*.py")
EXPECTED_EOL="lf"
ATTRIBUTES_FILE="${1:-}"

if [[ -z "$ATTRIBUTES_FILE" ]]; then
    if [[ -n "${BUILD_WORKSPACE_DIRECTORY:-}" ]]; then
        ATTRIBUTES_FILE="$BUILD_WORKSPACE_DIRECTORY/.gitattributes"
    elif [[ -n "${TEST_SRCDIR:-}" && -n "${TEST_WORKSPACE:-}" && -f "$TEST_SRCDIR/$TEST_WORKSPACE/.gitattributes" ]]; then
        ATTRIBUTES_FILE="$TEST_SRCDIR/$TEST_WORKSPACE/.gitattributes"
    elif [[ -n "${TEST_SRCDIR:-}" && -f "$TEST_SRCDIR/_main/.gitattributes" ]]; then
        ATTRIBUTES_FILE="$TEST_SRCDIR/_main/.gitattributes"
    else
        ATTRIBUTES_FILE=".gitattributes"
    fi
fi

if [[ ! -f "$ATTRIBUTES_FILE" ]]; then
    echo "missing .gitattributes" >&2
    exit 1
fi

for pattern in "${SCRIPT_PATTERNS[@]}"; do
    if ! awk -v pattern="$pattern" -v expected_eol="eol=$EXPECTED_EOL" '
        $1 == pattern {
            has_text = 0
            has_eol = 0
            for (field = 2; field <= NF; field++) {
                if ($field == "#") {
                    break
                }
                if ($field == "text") {
                    has_text = 1
                }
                if ($field == expected_eol) {
                    has_eol = 1
                }
            }
            if (has_text && has_eol) {
                found = 1
            }
        }
        END { exit found ? 0 : 1 }
    ' "$ATTRIBUTES_FILE"; then
        echo "missing required rule: $pattern text eol=lf" >&2
        exit 1
    fi
done

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    for pattern in "${SCRIPT_PATTERNS[@]}"; do
        while IFS= read -r script; do
            actual="$(git check-attr eol -- "$script" | awk -F': ' '{print $3}')"
            if [[ "$actual" != "$EXPECTED_EOL" ]]; then
                echo "$script resolves to eol=$actual, expected eol=$EXPECTED_EOL" >&2
                exit 1
            fi
        done < <(git ls-files "$pattern")
    done
fi
