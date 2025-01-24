load("//:third_party/org_swig_swig/repo.bzl", org_swig_swig = "repo")
load("//third_party/com_google_sentencepiece:repo.bzl", com_google_sentencepiece = "repo")

def _non_module_deps_impl(mctx):
    com_google_sentencepiece()
    org_swig_swig()

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

non_module_deps = module_extension(
    implementation = _non_module_deps_impl,
)
