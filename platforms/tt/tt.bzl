load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")

def _tt_impl(mctx):
    http_archive(
        name = "libpjrt_tt",
        build_file = "libpjrt_tt.BUILD.bazel",
        type = "zip",
        url = "https://pypi.eng.aws.tenstorrent.com/pjrt-plugin-tt/pjrt_plugin_tt-1.2.0.dev20260518002707-cp312-cp312-manylinux_2_34_x86_64.whl",
        sha256 = "219eee7227c043cb8ce13f69e14b9a1e5f6d8189523442d4d69c48d863bd5b5d",
        patch_cmds = [
            "find pjrt_plugin_tt/tt-metal -name 'BUILD.bazel' -delete",
        ],
    )

    http_archive(
        name = "libsfpi",
        build_file = "libsfpi.BUILD.bazel",
        strip_prefix = "sfpi",
        url = "https://github.com/tenstorrent/sfpi/releases/download/7.48.0/sfpi_7.48.0_x86_64_debian.txz",
        sha256 = "1bae34c2b9491f8a40b956c1a27c9421775d4a7d82d31578b47c05d4cb673d7e",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libpjrt_tt"],
        root_module_direct_dev_deps = [],
    )

tt_packages = module_extension(
    implementation = _tt_impl,
)
