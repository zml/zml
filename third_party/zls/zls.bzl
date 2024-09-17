load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

_VERSION = "0.13.0"

_ARCH = {
    "x86_64-linux": struct(
        sha256 = "ec4c1b45caf88e2bcb9ebb16c670603cc596e4f621b96184dfbe837b39cd8410",
        exec_compatible_with = [
            "@platforms//os:linux",
            "@platforms//cpu:x86_64",
        ],
    ),
    "aarch64-macos": struct(
        sha256 = "9848514524f5e5d33997ac280b7d92388407209d4b8d4be3866dc3cf30ca6ca8",
        exec_compatible_with = [
            "@platforms//os:macos",
            "@platforms//cpu:aarch64",
        ],
    ),
}

ZlsInfo = provider(
    fields = {
        "bin": "ZLS binary",
    },
)

def _zls_toolchain_impl(ctx):
    default = DefaultInfo(
        files = depset(direct = [ctx.file.zls]),
    )
    zlsinfo = ZlsInfo(
        bin = ctx.file.zls,
    )
    toolchain_info = platform_common.ToolchainInfo(
        default = default,
        zlsinfo = zlsinfo,
    )

    return [
        default,
        zlsinfo,
        toolchain_info,
    ]

zls_toolchain = rule(
    implementation = _zls_toolchain_impl,
    attrs = {
        "zls": attr.label(
            executable = True,
            allow_single_file = True,
            cfg = "exec",
        ),
    },
)

def _repo_impl(mctx):
    for arch, config in _ARCH.items():
        http_archive(
            name = "zls_{}".format(arch),
            url = "https://github.com/zigtools/zls/releases/download/{version}/zls-{arch}.tar.xz".format(
                version = _VERSION,
                arch = arch,
            ),
            sha256 = config.sha256,
            build_file_content = """\
load("@zml//third_party/zls:zls.bzl", "zls_toolchain")
zls_toolchain(name = "toolchain", zls = "zls")
""",
        )
    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

repo = module_extension(
    implementation = _repo_impl,
)

def targets():
    for arch, config in _ARCH.items():
        native.toolchain(
            name = "toolchain_{}".format(arch),
            exec_compatible_with = config.exec_compatible_with,
            target_compatible_with = config.exec_compatible_with,
            toolchain = "@zls_{}//:toolchain".format(arch),
            toolchain_type = "@zml//third_party/zls:toolchain_type",
        )

_ZIG_RUNNER_TPL = """\
#!/bin/bash

if [[ "${{1}}" == "build" ]]; then
    for arg in "${{@:2}}"; do
        if [[ "${{arg}}" == "-Dcmd="* ]]; then
            cd ${{BUILD_WORKSPACE_DIRECTORY}}
            exec ${{arg/-Dcmd=/}}
        fi
    done
fi

export ZIG_GLOBAL_CACHE_DIR="$(realpath {zig_cache})"
export ZIG_LOCAL_CACHE_DIR="$(realpath {zig_cache})"
export ZIG_LIB_DIR="$(realpath {zig_lib_path})"
exec {zig_exe_path} "${{@}}"
"""

_RUNNER_TPL = """\
#!/bin/bash
set -eo pipefail

zig() {{
    if [[ "${{1}}" == "build" ]]; then
        for arg in "${{@:2}}"; do
            if [[ "${{arg}}" == "-Dcmd="* ]]; then
                cd ${{BUILD_WORKSPACE_DIRECTORY}}
                exec ${{arg/-Dcmd=/}}
            fi
        done
    fi

    export ZIG_GLOBAL_CACHE_DIR="$(realpath {zig_cache})"
    export ZIG_LOCAL_CACHE_DIR="$(realpath {zig_cache})"
    export ZIG_LIB_DIR="$(realpath {zig_lib_path})"
    exec {zig_exe_path} "${{@}}"
}}

zls() {{
    json_config="$(mktemp)"
    ZLS_ARGS=("--config-path" "${{json_config}}")

    cat <<EOF > ${{json_config}}
{{
    "zig_lib_path": "$(realpath {zig_lib_path})",
    "zig_exe_path": "$(realpath {zig_exe_path})",
    "global_cache_path": "$(realpath {zig_cache})"
}}
EOF

    while ((${{#}})); do
        case "${{1}}" in
        --config-path)
            cat "${{2}}" >> "${{json_config}}"
            {jq} -s add "${{json_config}}" > "${{json_config}}.tmp"
            mv "${{json_config}}.tmp" "${{json_config}}"
            shift 2
            ;;
        *)
            ZLS_ARGS+=("${{1}}")
            shift
            ;;
        esac
    done

    exec {zls} "${{ZLS_ARGS[@]}}"
}}

case $1 in
    zig)
        shift
        zig "${{@}}"
        ;;
    zls)
        shift
        zls "${{@}}"
        ;;
esac
"""

def _zls_runner_impl(ctx):
    jqinfo = ctx.toolchains["@aspect_bazel_lib//lib:jq_toolchain_type"].jqinfo
    zigtoolchaininfo = ctx.toolchains["@rules_zig//zig:toolchain_type"].zigtoolchaininfo
    zlsinfo = ctx.toolchains["@zml//third_party/zls:toolchain_type"].zlsinfo

    zls_runner = ctx.actions.declare_file(ctx.label.name + ".zls_runner.sh")
    ctx.actions.write(zls_runner, _RUNNER_TPL.format(
        jq = jqinfo.bin.short_path,
        zig_cache = zigtoolchaininfo.zig_cache,
        zig_exe_path = zigtoolchaininfo.zig_exe.short_path,
        zig_lib_path = zigtoolchaininfo.zig_lib.short_path,
        zls = zlsinfo.bin.short_path,
    ))

    return [
        DefaultInfo(
            files = depset([zls_runner]),
            executable = zls_runner,
            runfiles = ctx.runfiles(
                files = [
                    ctx.executable.zig,
                    jqinfo.bin,
                    zlsinfo.bin,
                ],
                transitive_files = zigtoolchaininfo.zig_files,
            ),
        ),
    ]

zls_runner = rule(
    implementation = _zls_runner_impl,
    attrs = {
        "zig": attr.label(mandatory = True, executable = True, cfg = "exec"),
    },
    executable = True,
    toolchains = [
        "@rules_zig//zig:toolchain_type",
        "@aspect_bazel_lib//lib:jq_toolchain_type",
        "@zml//third_party/zls:toolchain_type",
    ],
)

def _zig_runner_impl(ctx):
    zigtoolchaininfo = ctx.toolchains["@rules_zig//zig:toolchain_type"].zigtoolchaininfo

    zig_runner = ctx.actions.declare_file(ctx.label.name + ".zig_runner.sh")
    ctx.actions.write(zig_runner, _ZIG_RUNNER_TPL.format(
        zig_cache = zigtoolchaininfo.zig_cache,
        zig_exe_path = zigtoolchaininfo.zig_exe.short_path,
        zig_lib_path = zigtoolchaininfo.zig_lib.short_path,
    ))

    return [
        DefaultInfo(
            files = depset([zig_runner]),
            executable = zig_runner,
            runfiles = ctx.runfiles(
                files = [zig_runner],
                transitive_files = zigtoolchaininfo.zig_files,
            ),
        ),
    ]

zig_runner = rule(
    implementation = _zig_runner_impl,
    executable = True,
    toolchains = ["@rules_zig//zig:toolchain_type"],
)
