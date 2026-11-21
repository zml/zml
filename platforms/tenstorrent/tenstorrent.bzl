"""Consume the complete Tenstorrent sandbox produced by XLA (//zml/tenstorrent:archive)."""

load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")

# PLACEHOLDER: the archive is not published on the mirror yet. Set both from the
# zml_plugins workflow output, as for Furiosa, e.g.
# "https://mirror.zml.ai/plugins/<build>/zml-tenstorrent-linux-amd64.tar.zst".
_MIRROR_URL = None
_MIRROR_SHA256 = None

# Absolute path of a locally built zml-tenstorrent-linux-amd64.tar.zst, passed as
#   --repo_env=ZML_TENSTORRENT_ARCHIVE=/path/to/zml-tenstorrent-linux-amd64.tar.zst
# It takes precedence over the mirror. The archive is watched: rebuilding it in
# place refetches the repository on the next build, with no edit here.
_LOCAL_ARCHIVE_ENV = "ZML_TENSTORRENT_ARCHIVE"

_BUILD_FILE = Label("//platforms/tenstorrent:libzml_tenstorrent.BUILD.bazel")

def _local_archive_impl(rctx):
    archive = rctx.path(rctx.attr.archive)
    rctx.watch(archive)
    if not archive.exists:
        fail("{}={} does not exist".format(_LOCAL_ARCHIVE_ENV, rctx.attr.archive))
    rctx.extract(archive)
    rctx.symlink(rctx.attr.build_file, "BUILD.bazel")

_local_archive = repository_rule(
    implementation = _local_archive_impl,
    attrs = {
        "archive": attr.string(mandatory = True),
        "build_file": attr.label(mandatory = True, allow_single_file = True),
    },
)

def _unpublished_impl(rctx):
    # Keeps queries over the whole graph working; only a build that enables
    # the platform needs the archive, and it fails with this message.
    rctx.file("BUILD.bazel", """\
load("@rules_cc//cc:cc_library.bzl", "cc_library")

genrule(
    name = "sandbox",
    outs = ["sandbox.missing"],
    cmd = "echo 'libzml_tenstorrent: no published archive yet; pass --repo_env={env}=/abs/path/zml-tenstorrent-linux-amd64.tar.zst' >&2; false",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "libzml_tenstorrent",
    data = [":sandbox"],
    visibility = ["//visibility:public"],
)
""".format(env = _LOCAL_ARCHIVE_ENV))

_unpublished = repository_rule(implementation = _unpublished_impl)

def _tenstorrent_impl(mctx):
    local_archive = mctx.getenv(_LOCAL_ARCHIVE_ENV)
    if local_archive:
        _local_archive(
            name = "libzml_tenstorrent",
            archive = local_archive,
            build_file = _BUILD_FILE,
        )
    elif _MIRROR_URL:
        http_archive(
            name = "libzml_tenstorrent",
            build_file = _BUILD_FILE,
            url = _MIRROR_URL,
            sha256 = _MIRROR_SHA256,
        )
    else:
        _unpublished(name = "libzml_tenstorrent")

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libzml_tenstorrent"],
        root_module_direct_dev_deps = [],
    )

tenstorrent_packages = module_extension(
    implementation = _tenstorrent_impl,
)
