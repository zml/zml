load("//third_party/arocc:repo.bzl", arocc = "repo")
load("//third_party/cloud_accelerator_diagnostics:repo.bzl", cloud_accelerator_diagnostics = "repo")
load("//third_party/com_google_sentencepiece:repo.bzl", com_google_sentencepiece = "repo")
load("//third_party/iree:repo.bzl", iree = "repo")
load("//third_party/mnist:repo.bzl", mnist = "repo")
load("//third_party/org_swig_swig:repo.bzl", org_swig_swig = "repo")
load("//third_party/pprof:repo.bzl", pprof = "repo")
load("//third_party/tracy:repo.bzl", tracy = "repo")
load("//third_party/translate-c:repo.bzl", translate_c = "repo")
load("//third_party/xla:repo.bzl", xla = "repo")
load("//third_party/flashattn:repo.bzl", flashattn = "repo")
load("//third_party/linenoise:repo.bzl", linenoise = "repo")
load("//third_party/stb:repo.bzl", stb = "repo")
load("//third_party/zigimg:repo.bzl", zigimg = "repo")
load("//third_party/uucode:repo.bzl", uucode = "repo")
load("//third_party/libvaxis:repo.bzl", libvaxis = "repo")

def _non_module_deps_impl(mctx):
    cloud_accelerator_diagnostics()
    com_google_sentencepiece()
    org_swig_swig()
    pprof()
    tracy()
    mnist()
    xla()
    arocc()
    translate_c()
    flashattn()
    linenoise()
    stb()
    zigimg()
    uucode()
    libvaxis()
    iree()

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

non_module_deps = module_extension(
    implementation = _non_module_deps_impl,
)
