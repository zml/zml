load("@rules_zig//zig:defs.bzl", "BINARY_KIND", "zig_binary")

def zig_cc_binary(name, args = None, env = None, data = [], deps = [], visibility = None, **kwargs):
    zig_binary(
        name = "{}_lib".format(name),
        kind = BINARY_KIND.static_lib,
        deps = deps + [
            "@rules_zig//zig/lib:libc",
        ],
        **kwargs
    )
    native.cc_binary(
        name = name,
        args = args,
        env = env,
        data = data,
        deps = [":{}_lib".format(name)],
        visibility = visibility,
    )

def zig_cc_test(name, env = None, data = [], deps = [], test_runner = None, visibility = None, **kwargs):
    zig_binary(
        name = "{}_test_lib".format(name),
        kind = BINARY_KIND.test_lib,
        test_runner = test_runner,
        data = data,
        deps = deps + [
            "@rules_zig//zig/lib:libc",
        ],
        **kwargs
    )
    native.cc_test(
        name = name,
        env = env,
        data = data,
        deps = [":{}_test_lib".format(name)],
        visibility = visibility,
        linkstatic = True,
    )
