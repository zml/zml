def _uucode_repo_impl(ctx):
    ctx.download_and_extract(
        url = "https://github.com/jacobsandlund/uucode/archive/{}.tar.gz".format(ctx.attr.commit),
        stripPrefix = "uucode-{}".format(ctx.attr.commit),
    )

    ctx.symlink(ctx.attr.build_config, "build_config.zig")
    ctx.symlink(ctx.attr.build_file, "BUILD.bazel")

_uucode_repo = repository_rule(
    implementation = _uucode_repo_impl,
    attrs = {
        "commit": attr.string(mandatory = True),
        "build_config": attr.label(mandatory = True, allow_single_file = True),
        "build_file": attr.label(mandatory = True, allow_single_file = True),
    },
)

def repo():
    _uucode_repo(
        name = "uucode",
        commit = "f1c687f09174f9f11eda14d3e3ce497d68ab09a6",
        build_config = "//third_party/uucode:build_config.zig",
        build_file = "//third_party/uucode:uucode.bazel",
    )
