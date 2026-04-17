load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")

def _tt_impl(mctx):
    http_archive(
        name = "libpjrt_tt",
        build_file = "libpjrt_tt.BUILD.bazel",
        type = "zip",
        url = "https://pypi.eng.aws.tenstorrent.com/pjrt-plugin-tt/pjrt_plugin_tt-1.1.0.dev20260417001937-cp312-cp312-manylinux_2_34_x86_64.whl",
        sha256 = "40c0d60f127adb0e2f048e5a41c4593c5673a764145976f0117e29c8e9691bd8",
        patch_cmds = [
            "find pjrt_plugin_tt/tt-metal -name 'BUILD.bazel' -delete",
        ],
    )

    http_archive(
        name = "libsfpi",
        build_file = "libsfpi.BUILD.bazel",
        strip_prefix = "sfpi",
        url = "https://github.com/tenstorrent/sfpi/releases/download/7.43.0/sfpi_7.43.0_x86_64_debian.txz",
        sha256 = "35ae9e904e1df76d56001cf27a5e8310e65b366cca5a92ef302ba7f05c35d08d",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libpjrt_tt"],
        root_module_direct_dev_deps = [],
    )

tt_packages = module_extension(
    implementation = _tt_impl,
)
