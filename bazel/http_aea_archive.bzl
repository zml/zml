"""Repository rule for downloading and extracting Apple Encrypted Archive assets on macOS and Linux AMD64/ARM64."""

load("@bazel_lib//lib:repo_utils.bzl", "repo_utils")

def _fail_result(step, result):
    if result.return_code != 0:
        fail("%s failed with exit code %d\nstdout:\n%s\nstderr:\n%s" % (
            step,
            result.return_code,
            result.stdout,
            result.stderr,
        ))

def _execute(rctx, step, argv, quiet = True):
    result = rctx.execute([str(arg) for arg in argv], quiet = quiet)
    _fail_result(step, result)
    return result

def _host_tools(rctx):
    platform = repo_utils.platform(rctx)
    if platform not in ["darwin_amd64", "darwin_arm64", "linux_amd64", "linux_arm64"]:
        fail("http_aea_archive supports macOS and Linux AMD64/ARM64 repository hosts; got %s/%s" % (rctx.os.name, rctx.os.arch))
    ipsw = Label("@ipsw_{}//:ipsw".format(platform)) if platform.startswith("linux_") else None
    sevenzip = Label("@sevenzip_{}//:7zz".format(platform))
    return (ipsw, sevenzip)

def _single_restore_dmg(restore_dirs):
    dmgs = []
    entries = []
    for restore_dir in restore_dirs:
        if not restore_dir.exists:
            continue
        for entry in restore_dir.readdir():
            entries.append(str(entry))
            if not entry.is_dir and entry.basename.endswith(".dmg"):
                dmgs.append(entry)

    if len(dmgs) != 1:
        fail("expected exactly one DMG under AssetData/Restore, got %d; entries: %s" % (
            len(dmgs),
            ", ".join(sorted(entries)),
        ))

    return dmgs[0]

def _linux_restore_dirs(rctx, ipsw, aar):
    # Create an explicit empty config. It avoids loading the user's ipsw configuration.
    rctx.file("ipsw-config.yaml", "{}\n")
    _execute(rctx, "ipsw ota extract", [
        rctx.path(ipsw),
        "--config",
        rctx.path("ipsw-config.yaml"),
        "ota",
        "extract",
        aar,
        "--key-val",
        rctx.attr.archive_decryption_key,
        "--pattern",
        "AssetData/Restore/.*\\.dmg$",  # pattern to match DMG files under AssetData/Restore
        "--output",
        rctx.path("ipsw-output"),
        "--confirm",
        "--no-color",
    ])
    output = rctx.path("ipsw-output")
    restore_dirs = []
    if output.exists:
        # ipsw prefixes extracted paths with an asset-specific directory.
        for entry in output.readdir():
            if entry.is_dir:
                restore_dirs.append(entry.get_child("AssetData", "Restore"))
    return restore_dirs

def _http_aea_archive_impl(rctx):
    ipsw, sevenzip = _host_tools(rctx)

    urls = []
    if rctx.attr.url:
        urls.append(rctx.attr.url)
    urls.extend(rctx.attr.urls)
    if not urls:
        fail("http_aea_archive requires url or urls")

    if not rctx.attr.sha256 and not rctx.attr.integrity:
        fail("http_aea_archive requires sha256 or integrity for reproducible downloads")
    if not rctx.attr.archive_decryption_key:
        fail("archive_decryption_key is required")

    aar = rctx.path("asset.aar")  # path to the downloaded Apple Encrypted Archive (AAR) file
    download_kwargs = {
        "canonical_id": rctx.attr.canonical_id,
        "integrity": rctx.attr.integrity,
        "output": aar,
        "sha256": rctx.attr.sha256,
        "url": urls,
    }
    if rctx.attr.netrc:
        download_kwargs["auth"] = rctx.use_netrc(
            rctx.read_netrc(rctx.attr.netrc),
            urls,
            rctx.attr.auth_patterns,
        )

    rctx.download(**download_kwargs)

    if ipsw:
        restore_dirs = _linux_restore_dirs(rctx, ipsw, aar)
    else:
        _execute(rctx, "aa patch", [
            "/usr/bin/aa",
            "patch",
            "-i",
            aar,
            "-key-value",
            "base64:%s" % rctx.attr.archive_decryption_key,
            "-src",
            "/var/empty",
            "-dst",
            rctx.path("."),
        ])
        restore_dirs = [rctx.path("AssetData/Restore")]
    dmg = _single_restore_dmg(restore_dirs)

    extract_args = [
        rctx.path(sevenzip),
        "x",
        "-y",
        dmg,
    ]
    extract_args.extend(rctx.attr.includes)
    _execute(rctx, "7z extract dmg", extract_args)

    # Central cleanup of temporary files and directories after extraction.
    # If absent, rctx.delete will handle it gracefully.
    rctx.delete("asset.aar")
    rctx.delete("AssetData")
    rctx.delete("ipsw-output")
    rctx.delete("ipsw-config.yaml")

    rctx.template("BUILD.bazel", rctx.attr.build_file)

    return None

http_aea_archive = repository_rule(
    implementation = _http_aea_archive_impl,
    attrs = {
        "url": attr.string(),
        "urls": attr.string_list(),
        "archive_decryption_key": attr.string(mandatory = True),
        "sha256": attr.string(),
        "integrity": attr.string(),
        "canonical_id": attr.string(),
        "netrc": attr.string(),
        "auth_patterns": attr.string_dict(),
        "build_file": attr.label(
            allow_single_file = True,
        ),
        "includes": attr.string_list(
            default = [],
            doc = "Patterns to pass to 7z for selective extraction from the DMG. If empty, extracts everything.",
        ),
    },
    doc = "Downloads and extracts an AEA-wrapped AppleArchive asset on macOS and Linux AMD64/ARM64.",
)
