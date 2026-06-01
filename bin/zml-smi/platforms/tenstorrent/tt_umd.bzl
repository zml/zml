load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

_VISIBILITY = """package(default_visibility = ["//visibility:public"])
"""

_FMT_BUILD = _VISIBILITY + """
cc_library(
    name = "fmt",
    hdrs = glob(["include/fmt/*.h"]),
    defines = ["FMT_HEADER_ONLY"],
    includes = ["include"],
)
"""

_SPDLOG_BUILD = _VISIBILITY + """
cc_library(
    name = "spdlog",
    hdrs = glob(["include/spdlog/**/*.h"]),
    defines = [
        "SPDLOG_HEADER_ONLY",
        "SPDLOG_FMT_EXTERNAL_HO",
    ],
    includes = ["include"],
)
"""

_TT_LOGGER_BUILD = _VISIBILITY + """
cc_library(
    name = "tt_logger",
    hdrs = glob(["include/**/*.hpp"]),
    includes = ["include"],
)
"""

_ASIO_BUILD = _VISIBILITY + """
cc_library(
    name = "asio",
    hdrs = glob([
        "asio/include/**/*.hpp",
        "asio/include/**/*.ipp",
    ]),
    defines = ["ASIO_STANDALONE"],
    includes = ["asio/include"],
)
"""

def _tt_umd_impl(mctx):
    new_git_repository(
        name = "fmt",
        remote = "https://github.com/fmtlib/fmt.git",
        commit = "123913715afeb8a437e6388b4473fcc4753e1c9a",  # 11.1.4
        build_file_content = _FMT_BUILD,
    )
    new_git_repository(
        name = "spdlog",
        remote = "https://github.com/gabime/spdlog.git",
        commit = "48bcf39a661a13be22666ac64db8a7f886f2637e",  # v1.15.2
        build_file_content = _SPDLOG_BUILD,
    )
    new_git_repository(
        name = "tt_logger",
        remote = "https://github.com/tenstorrent/tt-logger.git",
        commit = "359183f6549c65f8b1a4aee7940fa9d966c2b626",  # v1.1.8
        build_file_content = _TT_LOGGER_BUILD,
    )
    new_git_repository(
        name = "asio",
        remote = "https://github.com/chriskohlhoff/asio.git",
        commit = "12e0ce9e0500bf0f247dbd1ae894272656456079",  # asio-1-30-2
        build_file_content = _ASIO_BUILD,
    )
    new_git_repository(
        name = "tt_umd",
        remote = "https://github.com/tenstorrent/tt-umd.git",
        commit = "2f29bb1fe61c094331453c60047770d6e0b3bca5",  # v0.9.5-dev.260424
        build_file = "//bin/zml-smi/platforms/tenstorrent:tt_umd.BUILD.bazel",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["asio", "fmt", "spdlog", "tt_logger", "tt_umd"],
        root_module_direct_dev_deps = [],
    )

tt_umd_packages = module_extension(
    implementation = _tt_umd_impl,
)
