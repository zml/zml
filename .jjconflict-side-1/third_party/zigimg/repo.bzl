def _zigimg_repo_impl(ctx):
    ctx.download_and_extract(
        url = "https://github.com/zigimg/zigimg/archive/{}.tar.gz".format(ctx.attr.commit),
        stripPrefix = "zigimg-{}".format(ctx.attr.commit),
    )

    ctx.patch(ctx.attr.patch, strip = 1)
    ctx.symlink(ctx.attr.build_file, "BUILD.bazel")

_zigimg_repo = repository_rule(
    implementation = _zigimg_repo_impl,
    attrs = {
        "commit": attr.string(mandatory = True),
        "patch": attr.label(mandatory = True, allow_single_file = True),
        "build_file": attr.label(mandatory = True, allow_single_file = True),
    },
)

def repo():
    _zigimg_repo(
        name = "zigimg",
        commit = "7b98e82621fe302a9edc147df1191f4d1b7ff7a5",
        patch = "//third_party/zigimg:bump-zig.patch",
        build_file = "//third_party/zigimg:zigimg.bazel",
    )
