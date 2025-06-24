load("@aspect_bazel_lib//lib:tar.bzl", "mtree_spec", "tar")
load("@rules_zig//zig:defs.bzl", "zig_binary", "BINARY_KIND")

def zig_srcs(name, zig_bin="", zig_lib=""):
    """For a given zig_library, recursively extract all zig sources into a tarball.

    This also includes the files translated from C headers.
    It's also possible to pass zig_lib instead of zig_bin in which case,
    The rule takes care of creating an intermediary binary from the lib.
    """
    # TODO: this forces to build the test target which isn't needed.
    # The problem is we need the binary target to get the translate-c output.
    # What is missing from rules_zig is a "translate_c" target that we could call directly
    # without needing to build the corresponding binary.
    if zig_bin == "":
        zig_bin = "{}_bin".format(name)
        zig_binary(
            name = zig_bin,
            kind = BINARY_KIND.bc,
            tags = ["manual", "@rules_zig//zig/lib:libc"],
            deps = [zig_lib],
        )

    native.filegroup(
        name = "{}_files".format(name),
        srcs = [zig_bin],
        output_group = "srcs",
    )
    mtree_spec(
        name = "{}_mtree".format(name),
        srcs = [":{}_files".format(name)],
    )
    tar(
        name = name,
        srcs = ["{}_files".format(name)],
        args = [],
        mtree = "{}_mtree".format(name),
    )
