#!/bin/sh
set -eu

generated_source="$TEST_SRCDIR/$TEST_WORKSPACE/$1"

for expected in \
    'self.weight.dotAt(input, .d, .{' \
    ').addAt(self.bias, .{' \
    ').reluAt(.{' \
    'input.flattenAt(.{' \
    ').convertAt(.f32, .{' \
    'x.argMaxAt(0, .{' \
    '.indices.convertAt(.u8, .{'
do
    if ! grep -F "$expected" "$generated_source" >/dev/null; then
        echo "missing MNIST AST rewrite: $expected" >&2
        exit 1
    fi
done
