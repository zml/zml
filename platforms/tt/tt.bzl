load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")

def _tt_impl(mctx):
    http_archive(
        name = "libpjrt_tt",
        build_file = "libpjrt_tt.BUILD.bazel",
        type = "zip",
        url = "https://pypi.eng.aws.tenstorrent.com/pjrt-plugin-tt/pjrt_plugin_tt-1.2.0.dev20260525002927-cp312-cp312-manylinux_2_34_x86_64.whl",
        sha256 = "a4567bffaf2715634350277b4980b64d7d31dc670bf50d5075c3aee1c104b722",
        patch_cmds = [
            "find pjrt_plugin_tt/tt-metal -name 'BUILD.bazel' -delete",
        ],
    )

    http_archive(
        name = "libsfpi",
        build_file = "libsfpi.BUILD.bazel",
        strip_prefix = "sfpi",
        url = "https://github.com/tenstorrent/sfpi/releases/download/7.52.0/sfpi_7.52.0_x86_64_debian.txz",
        sha256 = "9da7af93ac12a1c7d05ffc60407924a4c093ffa200a041a77bf7af584bcde18b",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libpjrt_tt"],
        root_module_direct_dev_deps = [],
    )

tt_packages = module_extension(
    implementation = _tt_impl,
)
