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

def build_runner_tpl(target):
    return """\
const std = @import("std");

pub fn main() !void {{
    var gpa: std.heap.GeneralPurposeAllocator(.{{}}) = .{{}};
    defer _ = gpa.deinit();
    var arena_ = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena_.deinit();
    const arena = arena_.allocator();

    const build_workspace_directory = try std.process.getEnvVarOwned(arena, "BUILD_WORKSPACE_DIRECTORY");
    var child = std.process.Child.init(&.{{
        "bazel",
        "run",
        {target},
    }}, arena);
    child.stdin_behavior = .Ignore;
    child.stdout_behavior = .Inherit;
    child.stderr_behavior = .Inherit;
    child.cwd = build_workspace_directory;
    _ = try child.spawnAndWait();
}}
""".format(target = repr(target))

def runner_tpl(zls, zig_exe_path, zig_lib_path, zig_cache, build_runner):
    return """\
#!/bin/bash
set -eo pipefail

json_config="$(mktemp)"
cat <<EOF > ${{json_config}}
{{
    "build_runner_path": "$(realpath {build_runner})",
    "global_cache_path": "$(realpath {zig_cache})",
    "zig_exe_path": "$(realpath {zig_exe_path})",
    "zig_lib_path": "$(realpath {zig_lib_path})"
}}
EOF

exec {zls} --config-path "${{json_config}}" "${{@}}"
""".format(
        zig_lib_path = zig_lib_path,
        zig_exe_path = zig_exe_path,
        zig_cache = zig_cache,
        zls = zls,
        build_runner = build_runner,
    )

def _zls_runner_impl(ctx):
    zigtoolchaininfo = ctx.toolchains["@rules_zig//zig:toolchain_type"].zigtoolchaininfo
    zlsinfo = ctx.toolchains["@zml//third_party/zls:toolchain_type"].zlsinfo

    build_runner = ctx.actions.declare_file(ctx.label.name + ".build_runner.zig")
    ctx.actions.write(build_runner, build_runner_tpl(str(ctx.attr.target.label)))

    zls_runner = ctx.actions.declare_file(ctx.label.name + ".zls_runner.sh")
    ctx.actions.write(zls_runner, runner_tpl(
        zig_cache = zigtoolchaininfo.zig_cache,
        zig_exe_path = zigtoolchaininfo.zig_exe.short_path,
        zig_lib_path = zigtoolchaininfo.zig_lib.short_path,
        zls = zlsinfo.bin.short_path,
        build_runner = build_runner.short_path,
    ))

    return [
        DefaultInfo(
            files = depset([zls_runner]),
            executable = zls_runner,
            runfiles = ctx.runfiles(
                files = [
                    build_runner,
                    zlsinfo.bin,
                ],
                transitive_files = zigtoolchaininfo.zig_files,
            ),
        ),
    ]

zls_runner = rule(
    implementation = _zls_runner_impl,
    attrs = {
        "target": attr.label(mandatory = True, executable = True, cfg = "exec"),
    },
    executable = True,
    toolchains = [
        "@rules_zig//zig:toolchain_type",
        "@zml//third_party/zls:toolchain_type",
    ],
)
