load("@bazel_lib//lib:repo_utils.bzl", "repo_utils")
load(
    "@bazel_tools//tools/build_defs/repo:cache.bzl",
    "DEFAULT_CANONICAL_ID_ENV",
    "get_default_canonical_id",
)
load(
    "@bazel_tools//tools/build_defs/repo:utils.bzl",
    "get_auth",
    "patch",
    "update_attrs",
    "workspace_and_buildfile",
)

def _get_source_urls(rctx):
    if not rctx.attr.url and not rctx.attr.urls:
        fail("At least one of url and urls must be provided")

    source_urls = []
    if rctx.attr.urls:
        source_urls = rctx.attr.urls
    if rctx.attr.url:
        source_urls = [rctx.attr.url] + source_urls
    return source_urls

def _prefixed_pattern(strip_prefix, pattern):
    if not strip_prefix:
        return pattern
    if not pattern:
        return strip_prefix
    if pattern == strip_prefix or pattern.startswith(strip_prefix + "/"):
        return pattern
    return strip_prefix + "/" + pattern

def _downloaded_archive_name(rctx, source_urls):
    if rctx.attr.type:
        return ".downloaded.archive." + rctx.attr.type

    basename = source_urls[0].split("?", 1)[0].split("/")[-1]
    if basename:
        return ".downloaded." + basename
    return ".downloaded.archive"

def _placeholder_dir_from_exclude(pattern):
    if pattern.endswith("/*"):
        return pattern[:-2].strip("/")
    return None

def _ensure_placeholder_dirs(rctx, add_prefix):
    for exclude in rctx.attr.excludes:
        directory = _placeholder_dir_from_exclude(exclude)
        if not directory:
            continue
        if add_prefix:
            directory = add_prefix + "/" + directory
        keep_file = directory + "/.http_deb_archive.keep"
        rctx.file(keep_file, "")
        rctx.delete(keep_file)

def _host_bsdtar_label(rctx):
    platform = repo_utils.platform(rctx)
    binary = "tar.exe" if platform.startswith("windows_") else "tar"
    return Label("@bsd_tar_toolchains_{}//:{}".format(platform, binary))

def _update_integrity_attrs(rctx, attrs, archive_info, remote_files_info, remote_patches_info):
    integrity_override = {}
    if not rctx.attr.sha256 and not rctx.attr.integrity:
        integrity_override["integrity"] = archive_info.integrity

    remote_file_integrity = {path: info.integrity for path, info in remote_files_info.items()}
    if rctx.attr.remote_file_integrity != remote_file_integrity:
        integrity_override["remote_file_integrity"] = remote_file_integrity

    # TODO(cerisier): Remove this "if" when we no longer support bazel < 8.5.0
    if remote_patches_info:
        remote_patch_integrity = {url: info.integrity for url, info in remote_patches_info.items()}
        if rctx.attr.remote_patches != remote_patch_integrity:
            integrity_override["remote_patches"] = remote_patch_integrity

    if not integrity_override:
        return rctx.repo_metadata(reproducible = True)

    return rctx.repo_metadata(
        attrs_for_reproducibility = update_attrs(rctx.attr, attrs.keys(), integrity_override),
    )

# TODO(cerisier): Remove when we no longer support bazel < 8.5.0
def symlink_files(ctx):
    # type: (repository_ctx) -> None
    """Utility function for symlinking local files.

    This is intended to be used in the implementation function of a repository rule. It assumes the
    parameter `files` is present in `ctx.attr`.

    Existing files will be overwritten.

    Args:
      ctx: The repository context of the repository rule calling this utility
        function.
    """
    for path, label in ctx.attr.files.items():
        src_path = ctx.path(label)

        # On Windows `ctx.symlink` may be implemented as a copy, so the file MUST be watched
        ctx.watch(src_path)
        if not src_path.exists:
            fail("Input %s does not exist" % label)
        if ctx.path(path).exists:
            ctx.delete(path)
        ctx.symlink(src_path, path)

# TODO(cerisier): Remove when we no longer support bazel < 8.5.0
def download_remote_files(ctx, auth = None):
    """Utility function for downloading remote files.

    This rule is intended to be used in the implementation function of
    a repository rule. It assumes the parameters `remote_file_urls` and
    `remote_file_integrity` to be present in `ctx.attr`.

    Existing files will be overwritten.

    Args:
      ctx: The repository context of the repository rule calling this utility
        function.
      auth: An optional dict specifying authentication information for some of the URLs.

    Returns:
        dict mapping file paths to a download info.
    """
    pending = {
        path: ctx.download(
            remote_file_urls,
            path,
            canonical_id = ctx.attr.canonical_id,
            auth = get_auth(ctx, remote_file_urls) if auth == None else auth,
            integrity = ctx.attr.remote_file_integrity.get(path, ""),
            block = False,
            # Overlaid files may be shell scripts.
            executable = True,
        )
        for path, remote_file_urls in ctx.attr.remote_file_urls.items()
    }

    # Wait until the requests are done
    return {path: token.wait() for path, token in pending.items()}

def _http_deb_archive_impl(rctx):
    source_urls = _get_source_urls(rctx)

    if rctx.attr.build_file and rctx.attr.build_file_content:
        fail("Only one of build_file and build_file_content can be provided.")

    archive = _downloaded_archive_name(rctx, source_urls)
    archive_info = rctx.download(
        source_urls,
        archive,
        rctx.attr.sha256,
        canonical_id = rctx.attr.canonical_id or get_default_canonical_id(rctx, source_urls),
        auth = get_auth(rctx, source_urls),
        integrity = rctx.attr.integrity,
    )

    host_bsdtar = _host_bsdtar_label(rctx)
    res = rctx.execute([
        rctx.path(host_bsdtar),
        "-xf",
        archive,
        "--include=data.tar.*",
    ])
    for ext in ["gz", "xz", "zst"]:
        data = "data.tar.{}".format(ext)
        if rctx.path(data).exists:
            archive = data
            break

    strip_prefix = rctx.attr.strip_prefix.strip("/")
    includes = [_prefixed_pattern(strip_prefix, include) for include in rctx.attr.includes]
    excludes = [_prefixed_pattern(strip_prefix, exclude) for exclude in rctx.attr.excludes]

    if strip_prefix and not includes:
        includes = [strip_prefix, strip_prefix + "/*"]

    args = []
    for include in includes:
        args.extend(["--include", include])
    for exclude in excludes:
        args.extend(["--exclude", exclude])
    if strip_prefix:
        args.extend(["--strip-components", str(len(strip_prefix.split("/")))])
    args.extend(rctx.attr.bsdtar_extra_args)

    add_prefix = rctx.attr.add_prefix.strip("/")
    keep_file = None
    if add_prefix:
        keep_file = add_prefix + "/.http_deb_archive.keep"
        rctx.file(keep_file, "")
        args.extend(["-C", add_prefix])

    args.extend(["-xf", archive])
    res = rctx.execute([rctx.path(host_bsdtar)] + args)
    if res.return_code != 0:
        fail("Failed to extract archive: {}\n{}".format(res.stderr, res.stdout))

    if keep_file:
        rctx.delete(keep_file)
    _ensure_placeholder_dirs(rctx, add_prefix)
    rctx.delete(archive)

    workspace_and_buildfile(rctx)
    remote_files_info = download_remote_files(rctx)
    remote_patches_info = patch(rctx)
    symlink_files(rctx)

    return _update_integrity_attrs(rctx, _http_deb_archive_attrs, archive_info, remote_files_info, remote_patches_info)

_http_deb_archive_attrs = {
    "url": attr.string(),
    "urls": attr.string_list(),
    "sha256": attr.string(),
    "integrity": attr.string(),
    "netrc": attr.string(),
    "auth_patterns": attr.string_dict(),
    "canonical_id": attr.string(),
    "strip_prefix": attr.string(),
    "add_prefix": attr.string(default = ""),
    "files": attr.string_keyed_label_dict(default = {}),
    "type": attr.string(),
    "patches": attr.label_list(default = []),
    "remote_file_urls": attr.string_list_dict(default = {}),
    "remote_file_integrity": attr.string_dict(default = {}),
    "remote_module_file_urls": attr.string_list(default = []),
    "remote_module_file_integrity": attr.string(default = ""),
    "remote_patches": attr.string_dict(default = {}),
    "remote_patch_strip": attr.int(default = 0),
    "patch_tool": attr.string(default = ""),
    "patch_args": attr.string_list(default = []),
    "patch_strip": attr.int(default = 0),
    "patch_cmds": attr.string_list(default = []),
    "patch_cmds_win": attr.string_list(default = []),
    "build_file": attr.label(allow_single_file = True),
    "build_file_content": attr.string(),
    "workspace_file": attr.label(),
    "workspace_file_content": attr.string(),
    "includes": attr.string_list(default = []),
    "excludes": attr.string_list(default = []),
    "bsdtar_extra_args": attr.string_list(default = []),
}

http_deb_archive = repository_rule(
    implementation = _http_deb_archive_impl,
    attrs = _http_deb_archive_attrs,
    environ = [DEFAULT_CANONICAL_ID_ENV],
)
