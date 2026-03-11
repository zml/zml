def _uucode_repo_impl(ctx):
    ctx.download_and_extract(
        url = "https://github.com/jacobsandlund/uucode/archive/{}.tar.gz".format(ctx.attr.commit),
        stripPrefix = "uucode-{}".format(ctx.attr.commit),
    )

    ctx.symlink(ctx.attr.tables, "src/tables.zig")
    ctx.patch(ctx.attr.patch, strip = 1)
    ctx.symlink(ctx.attr.build_file, "BUILD.bazel")

_uucode_repo = repository_rule(
    implementation = _uucode_repo_impl,
    attrs = {
        "commit": attr.string(mandatory = True),
        "tables": attr.label(mandatory = True, allow_single_file = True),
        "patch": attr.label(mandatory = True, allow_single_file = True),
        "build_file": attr.label(mandatory = True, allow_single_file = True),
    },
)

def repo():
    _uucode_repo(
        name = "uucode",
        commit = "faab8894d753207878283d523add01817e23b805",
        tables = "//third_party/uucode:tables.zig",
        patch = "//third_party/uucode:fix-imports.patch",
        build_file = "//third_party/uucode:uucode.bazel",
    )
