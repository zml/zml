"""Repository rule for macOS Apple Encrypted Archive."""

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

def _require_macos(rctx):
    os_name = rctx.os.name.lower()
    if "mac" not in os_name and "darwin" not in os_name:
        fail("http_aea_archive is macOS-only; current repository OS is %r" % rctx.os.name)

def _single_restore_dmg(rctx):
    restore_dir = rctx.path("AssetData/Restore")
    if not restore_dir.exists:
        fail("aa patch did not produce AssetData/Restore")

    dmgs = []
    entries = []
    for entry in restore_dir.readdir():
        entries.append(entry.basename)
        if not entry.is_dir and entry.basename.endswith(".dmg"):
            dmgs.append(entry)

    if len(dmgs) != 1:
        fail("expected exactly one restore DMG in AssetData/Restore, got %d; entries: %s" % (
            len(dmgs),
            ", ".join(sorted(entries)),
        ))

    return dmgs[0]

def _http_aea_archive_impl(rctx):
    _require_macos(rctx)

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

    aar = rctx.path("asset.aar")
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
    rctx.delete("asset.aar")

    dmg = _single_restore_dmg(rctx)

    extract_args = [
        rctx.path(rctx.attr._sevenzip),
        "x",
        "-y",
        dmg,
    ]
    extract_args.extend(rctx.attr.includes)
    _execute(rctx, "7z extract dmg", extract_args)
    rctx.delete("AssetData")

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
        "_sevenzip": attr.label(
            allow_single_file = True,
            default = Label("@sevenzip_macos//:7zz"),
        ),
    },
    doc = "Downloads and extracts a macOS AEA-wrapped AppleArchive asset",
)
