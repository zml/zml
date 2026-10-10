load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")

def _tt_impl(mctx):
    # https://pypi.eng.aws.tenstorrent.com/pjrt-plugin-tt/
    http_archive(
        name = "libpjrt_tt",
        build_file = "libpjrt_tt.BUILD.bazel",
        type = "zip",
        url = "https://pypi.eng.aws.tenstorrent.com/pjrt-plugin-tt/pjrt_plugin_tt-1.5.0.dev20260831000501-cp312-cp312-manylinux_2_34_x86_64.whl",
        sha256 = "bfa7dd0834abb55cf37bf2fe57214d0a1074738155175cd0e177b6e1d5408bc1",
        patch_cmds = [
            "find pjrt_plugin_tt/tt-metal -name 'BUILD.bazel' -delete",
        ],
    )

    # Must match `pjrt_plugin_tt/tt-metal/tt_metal/sfpi-version` inside the wheel.
    http_archive(
        name = "libsfpi",
        build_file = "libsfpi.BUILD.bazel",
        strip_prefix = "sfpi",
        url = "https://github.com/tenstorrent/sfpi/releases/download/7.69.0/sfpi_7.69.0_x86_64_debian.txz",
        sha256 = "b0c93362de2f69b0e4c335abd126ff59748f5dae316980d17aa1525ed523eb14",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libpjrt_tt"],
        root_module_direct_dev_deps = [],
    )

tt_packages = module_extension(
    implementation = _tt_impl,
)
