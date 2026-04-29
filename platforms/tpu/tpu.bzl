load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")

def _tpu_impl(mctx):
    # https://storage.googleapis.com/jax-releases/libtpu_releases.html
    http_archive(
        name = "libpjrt_tpu",
        url = "https://files.pythonhosted.org/packages/42/7f/cc3ad9e9b866c701e90e6d8d4e332557fefb1f3ad1bebd9914ff09778691/libtpu-0.0.40-cp314-cp314t-manylinux_2_31_x86_64.whl",
        type = "zip",
        sha256 = "a1d01214bffe5a0910057014eaa17ed838eebc780e1538b4bd118684908ea120",
        build_file = "libpjrt_tpu.BUILD.bazel",
    )
    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libpjrt_tpu"],
        root_module_direct_dev_deps = [],
    )

tpu_packages = module_extension(
    implementation = _tpu_impl,
)
