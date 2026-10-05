def _uucode_repo_impl(ctx):
    ctx.download_and_extract(
        url = "https://github.com/jacobsandlund/uucode/archive/{}.tar.gz".format(ctx.attr.commit),
        stripPrefix = "uucode-{}".format(ctx.attr.commit),
    )
    ctx.patch(ctx.attr.config_storage_patch, strip = 1)

    ctx.symlink(ctx.attr.build_config, "build_config.zig")
    ctx.symlink(ctx.attr.build_file, "BUILD.bazel")

_uucode_repo = repository_rule(
    implementation = _uucode_repo_impl,
    attrs = {
        "commit": attr.string(mandatory = True),
        "build_config": attr.label(mandatory = True, allow_single_file = True),
        "build_file": attr.label(mandatory = True, allow_single_file = True),
        "config_storage_patch": attr.label(mandatory = True, allow_single_file = True),
    },
)

def repo():
    _uucode_repo(
        name = "uucode",
        # main branch
        commit = "1fb73433bba5d93366c57f23ff2e9d7939746500",
        build_config = "//third_party/uucode:build_config.zig",
        build_file = "//third_party/uucode:uucode.bazel",
        config_storage_patch = "//third_party/uucode:config-storage-module.patch",
    )
