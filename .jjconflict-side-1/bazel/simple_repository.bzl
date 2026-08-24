_UNSET = "_UNSET"

def _simple_repository_impl(rctx):
    if (rctx.attr.build_file == None) == (rctx.attr.build_file_content == _UNSET):
        fail("exactly one of `build_file` and `build_file_content` must be specified")

    if rctx.attr.build_file != None:
        # Remove any existing BUILD.bazel in the repository to ensure
        # the symlink to the defined build_file doesn't fail.
        rctx.delete("BUILD.bazel")
        rctx.symlink(rctx.attr.build_file, "BUILD.bazel")
    else:
        rctx.file("BUILD.bazel", rctx.attr.build_file_content)

simple_repository = repository_rule(
    implementation = _simple_repository_impl,
    attrs = {
        "build_file": attr.label(allow_single_file = True),
        "build_file_content": attr.string(default = _UNSET),
    },
    doc = "Makes an empty repository from just one BUILD.bazel file.",
    local = True,
)
