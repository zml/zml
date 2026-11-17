"""Analysis tests for runtime membership and the 0.1.0 consumer API."""

load("@bazel_skylib//lib:unittest.bzl", "analysistest", "asserts")
load(":fixtures.bzl", "GroupReportInfo")

def _groups_test_impl(ctx):
    env = analysistest.begin(ctx)
    result = analysistest.target_under_test(env)[GroupReportInfo]
    asserts.equals(env, ctx.attr.expected, result.groups)
    union = {entry: True for entries in result.groups.values() for entry in entries}
    asserts.equals(env, result.original, sorted(union.keys()))
    asserts.equals(env, ctx.attr.executable_group or None, result.executable_group)
    return analysistest.end(env)

groups_test = analysistest.make(
    _groups_test_impl,
    attrs = {
        "expected": attr.string_list_dict(),
        "executable_group": attr.string(),
    },
)

def _invalid_test_impl(ctx):
    env = analysistest.begin(ctx)
    asserts.expect_failure(env, ctx.attr.message)
    return analysistest.end(env)

invalid_test = analysistest.make(
    _invalid_test_impl,
    expect_failure = True,
    attrs = {"message": attr.string()},
)

def contents(name, executable = False):
    """Expected fixture runfiles, including both kinds of aliases."""
    return sorted([
        "file:" + name + ".txt",
        "root_symlink:" + name + "/root_alias=" + name + ("" if executable else ".txt"),
        "symlink:" + name + "/alias=" + name + ("" if executable else ".txt"),
    ] + (["file:" + name] if executable else []))
