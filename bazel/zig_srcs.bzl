load("@aspect_bazel_lib//lib:tar.bzl", "mtree_spec", "tar")
load("@bazel_tools//tools/build_defs/repo:filegroup.bzl", "filegroup")

def zig_srcs(name, zig_bin):
    """For a given zig_binary, recursivel extract all zig sources into a tarball.

    This also includes the files translated from C headers.
    """
    filegroup(
        name = "{}_zig_srcs".format(name),
        srcs = [zig_bin],
        output_group = "srcs",
    )
    mtree_spec(
        name = "{}_mtree".format(name),
        srcs = [":{}_zig_srcs".format(name)],
    )
    tar(
        name = name,
        srcs = ["{}_zig_srcs".format(name)],
        args = [],
        mtree = "{}_mtree".format(name),
    )
