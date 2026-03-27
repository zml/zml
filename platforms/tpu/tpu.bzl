load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")

def _tpu_impl(mctx):
    http_archive(
        name = "libpjrt_tpu",
        url = "https://files.pythonhosted.org/packages/ef/fc/e512372c4634d22ba10eaa778437bfe230d9a47fc856d6da5c644724278b/libtpu-0.0.38-cp314-cp314t-manylinux_2_31_x86_64.whl",
        type = "zip",
        sha256 = "48fd24bdf45156502c908296a30f8bb35d81ee0fafa1e54495f363fcac4d7814",
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
