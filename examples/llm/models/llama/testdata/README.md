# Llama regression evidence

This directory contains manual reproductions, not automated Bazel fixtures.
`session-position-plus-one.patch` is a historical negative reproduction: it
advances decode position incorrectly to check whether session comparisons detect
the error. It is not applied by the build and may need rebasing before use.

Obsolete machine-specific logs, process/resource dumps, JSON measurements and
reports were removed after checking code/build consumers and documentation links.
Git history retains those captures. They documented earlier packed-weight and
backend experiments, not acceptance results for the current typed backend.
Reusable diagnostics remain in `llama_tests.zig`; architecture and numerical
criteria are in [WHOLE_FORWARD.md](../WHOLE_FORWARD.md). Current validation is
recorded in [the Furiosa report](../../../../../platforms/furiosa/VALIDATION.md).
