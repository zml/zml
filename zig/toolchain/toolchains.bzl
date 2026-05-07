load("@rules_zig//zig:toolchain.bzl", "zig_target_toolchain", "zig_toolchain")

_TOOLCHAINS = {
    "linux_x86_64": {
        "target_compatible_with": [
            "@platforms//cpu:x86_64",
            "@platforms//os:linux",
        ],
    },
    "linux_aarch64": {
        "target_compatible_with": [
            "@platforms//cpu:aarch64",
            "@platforms//os:linux",
        ],
    },
    "macos_aarch64": {
        "target_compatible_with": [
            "@platforms//cpu:aarch64",
            "@platforms//os:macos",
        ],
    },
}

_TARGET_TOOLCHAINS = {
    # TODO "nvptx64-cuda-none": {},
}

def declare_toolchains():
    for name, attrs in _TOOLCHAINS.items():
        native.toolchain(
            name = "zig-{}-toolchain".format(name),
            target_settings = [
                "@rules_zig//zig/settings:translate_c_enabled",
            ],
            toolchain = ":zig_from_source",
            toolchain_type = "@rules_zig//zig:toolchain_type",
            **attrs,
        )
    for target, attrs in _TARGET_TOOLCHAINS.items():
        zig_target_toolchain(
            name = "zig-{}-zig_target_toolchain".format(target),
            target = target,
        )
        native.toolchain(
            name = "zig-{}-target_toolchain".format(target),
            toolchain = ":zig-{}-zig_target_toolchain".format(target),
            toolchain_type = "@rules_zig//zig/target:toolchain_type",
            **attrs,
        )
