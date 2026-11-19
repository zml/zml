# Cleanup: keep the archive and unpack rules. The marked URL override is temporary.
"""Install the MUSA SDK and PJRT plugin as a single relocatable runfile."""

def _musa_archive_impl(rctx):
    # Cleanup: replace the environment fallback and error with rctx.attr.url
    # once MODULE.bazel pins the published XLA archive before merging.
    url = rctx.attr.url or rctx.getenv("ZML_MUSA_ARCHIVE_URL")
    if not url:
        fail("Pass --repo_env=ZML_MUSA_ARCHIVE_URL=file:///absolute/path/to/the/XLA/5.1.0/archive.tar.zst")
    rctx.download(url, "archive.tar.zst", sha256 = rctx.attr.sha256)
    rctx.symlink(rctx.attr._build_file, "BUILD.bazel")

musa_archive = repository_rule(
    implementation = _musa_archive_impl,
    attrs = {
        "url": attr.string(),
        "sha256": attr.string(mandatory = True),
        "_build_file": attr.label(default = ":libzml_musa.BUILD.bazel"),
    },
)

def _musa_sandbox_impl(ctx):
    directory = ctx.actions.declare_directory(ctx.label.name)
    tar = ctx.toolchains["@tar.bzl//tar/toolchain:type"].tarinfo
    ctx.actions.run_shell(
        command = 'mkdir -p "$3" && exec "$1" -xf "$2" -C "$3"',
        arguments = [tar.binary.path, ctx.file.src.path, directory.path],
        tools = [tar.binary],
        inputs = [ctx.file.src],
        outputs = [directory],
        env = tar.default_env,
        mnemonic = "UnpackMusaSdk",
    )
    return [DefaultInfo(files = depset([directory]), runfiles = ctx.runfiles(files = [directory]))]

musa_sandbox = rule(
    implementation = _musa_sandbox_impl,
    attrs = {"src": attr.label(allow_single_file = True, mandatory = True)},
    toolchains = ["@tar.bzl//tar/toolchain:type"],
)
