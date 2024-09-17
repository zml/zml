load(
    "@bazel_tools//tools/build_defs/repo:utils.bzl",
    "get_auth",
    "patch",
    "workspace_and_buildfile",
)

def _http_deb_archive_impl(rctx):
    if rctx.attr.build_file and rctx.attr.build_file_content:
        fail("Only one of build_file and build_file_content can be provided.")
    download_info = rctx.download_and_extract(
        url = rctx.attr.urls,
        output = "tmp",
        sha256 = rctx.attr.sha256,
        type = "deb",
        stripPrefix = "",
        canonical_id = " ".join(rctx.attr.urls),
        auth = get_auth(rctx, rctx.attr.urls),
    )

    for ext in ["gz", "xz", "zst"]:
        data = "tmp/data.tar.{}".format(ext)
        if rctx.path(data).exists:
            rctx.extract(
                archive = data,
                output = "",
                stripPrefix = rctx.attr.strip_prefix,
            )
            rctx.delete("tmp")
            break
    workspace_and_buildfile(rctx)
    patch(rctx)

http_deb_archive = repository_rule(
    _http_deb_archive_impl,
    attrs = {
        "urls": attr.string_list(mandatory = True),
        "sha256": attr.string(mandatory = True),
        "strip_prefix": attr.string(),
        "build_file": attr.label(allow_single_file = True),
        "build_file_content": attr.string(),
        "workspace_file": attr.label(allow_single_file = True),
        "workspace_file_content": attr.string(),
    },
)
