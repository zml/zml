"""This module implements an alias rule to the resolved toolchain."""

DOC = """\
Exposes a concrete toolchain which is the result of Bazel resolving the
toolchain for the execution or target platform.
Workaround for https://github.com/bazelbuild/bazel/issues/14009
"""

# Forward all the providers
def _resolved_toolchain_impl(ctx):
    toolchain_info = ctx.toolchains["//third_party/zls:toolchain_type"]
    return [
        toolchain_info,
        toolchain_info.default,
        toolchain_info.zlstoolchaininfo,
    ]

# Copied from java_toolchain_alias
# https://cs.opensource.google/bazel/bazel/+/master:tools/jdk/java_toolchain_alias.bzl
resolved_toolchain = rule(
    implementation = _resolved_toolchain_impl,
    toolchains = ["//third_party/zls:toolchain_type"],
    doc = DOC,
)
