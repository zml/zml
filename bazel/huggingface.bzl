load(
    "@bazel_tools//tools/build_defs/repo:utils.bzl",
    "patch",
    "workspace_and_buildfile",
)

TREE_URL_TEMPLATE = "https://huggingface.co/api/models/{model}/tree/{commit}/{path}"
RAW_FILE_URL_REMPLATE = "https://huggingface.co/{model}/raw/{commit}/{path}"
LFS_FILE_URL_TEMPLATE = "https://huggingface.co/{model}/resolve/{commit}/{path}"

def _glob(rctx, str, patterns):
    cmd = "\n".join([
        """[[ "{str}" = {pattern} ]] && exit 0""".format(str = str, pattern = pattern)
        for pattern in patterns
    ] + ["exit 1"])
    return rctx.execute(["bash", "-c", cmd]).return_code == 0

def _ls(rctx, headers, path):
    url = TREE_URL_TEMPLATE.format(
        model = rctx.attr.model,
        commit = rctx.attr.commit,
        path = path,
    )
    rctx.download(url, path + ".index.json", headers = headers)
    ret = json.decode(rctx.read(path + ".index.json"))
    rctx.delete(path + ".index.json")
    return ret

def _get_token_via_env(rctx):
    return rctx.getenv("HUGGINGFACE_TOKEN")

def _get_token_via_file(rctx):
    p = rctx.path(rctx.getenv("HOME") + "/.cache/huggingface/token")
    if p.exists:
        return rctx.read(p)

def _get_token_via_git_credentials(rctx):
    input = """\
protocol=https
host=huggingface.co

"""
    res = rctx.execute(["bash", "-c", "echo '{}' | git credential fill".format(input)])
    if res.return_code != 0:
        return None
    for line in res.stdout.split("\n"):
        if line.startswith("password="):
            return line[len("password="):]
    return None

def _get_token(rctx):
    t = _get_token_via_env(rctx) or \
        _get_token_via_file(rctx) or \
        _get_token_via_git_credentials(rctx)
    if t:
        return t.strip()

def _huggingface_repository_impl(rctx):
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
    }

    token = _get_token(rctx)
    if token:
        headers["Authorization"] = "Bearer " + token

    includes = rctx.attr.includes
    excludes = rctx.attr.excludes
    stack = [""]
    downloads = []

    for _ in range(9999999):
        if (not stack):
            break
        path = stack.pop()
        for entry in _ls(rctx, headers, path):
            if entry["type"] == "directory":
                stack.append(entry["path"])
            elif entry["type"] == "file":
                if (excludes and _glob(rctx, entry["path"], excludes)):
                    continue
                if (not includes or _glob(rctx, entry["path"], includes)):
                    tpl = RAW_FILE_URL_REMPLATE
                    if ("lfs" in entry):
                        tpl = LFS_FILE_URL_TEMPLATE
                    url = tpl.format(
                        model = rctx.attr.model,
                        commit = rctx.attr.commit,
                        path = entry["path"],
                    )
                    downloads.append(rctx.download(
                        url = url,
                        output = entry["path"],
                        canonical_id = entry["oid"],
                        headers = headers,
                        block = False,
                    ))

    for download in downloads:
        download.wait()

    workspace_and_buildfile(rctx)
    patch(rctx)

huggingface_repository = repository_rule(
    implementation = _huggingface_repository_impl,
    attrs = {
        "model": attr.string(mandatory = True),
        "commit": attr.string(mandatory = True),
        "includes": attr.string_list(default = []),
        "excludes": attr.string_list(default = []),
        "patches": attr.label_list(),
        "patch_tool": attr.string(default = ""),
        "patch_args": attr.string_list(default = ["-p0"]),
        "patch_cmds": attr.string_list(default = []),
        "patch_cmds_win": attr.string_list(default = []),
        "build_file": attr.label(allow_single_file = True),
        "build_file_content": attr.string(),
        "workspace_file": attr.label(allow_single_file = True),
        "workspace_file_content": attr.string(),
    },
)

def _huggingface_impl(mctx):
    for mod in mctx.modules:
        for model in mod.tags.model:
            huggingface_repository(
                name = model.name,
                model = model.model,
                commit = model.commit,
                includes = model.includes,
                excludes = model.excludes,
                patches = model.patches,
                patch_tool = model.patch_tool,
                patch_args = model.patch_args,
                patch_cmds = model.patch_cmds,
                patch_cmds_win = model.patch_cmds_win,
                build_file = model.build_file,
                build_file_content = model.build_file_content,
                workspace_file = model.workspace_file,
                workspace_file_content = model.workspace_file_content,
            )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

huggingface = module_extension(
    implementation = _huggingface_impl,
    tag_classes = {
        "model": tag_class(
            attrs = {
                "name": attr.string(mandatory = True),
                "model": attr.string(mandatory = True),
                "commit": attr.string(mandatory = True),
                "includes": attr.string_list(default = []),
                "excludes": attr.string_list(default = []),
                "patches": attr.label_list(),
                "patch_tool": attr.string(default = ""),
                "patch_args": attr.string_list(default = ["-p0"]),
                "patch_cmds": attr.string_list(default = []),
                "patch_cmds_win": attr.string_list(default = []),
                "build_file": attr.label(allow_single_file = True),
                "build_file_content": attr.string(),
                "workspace_file": attr.label(allow_single_file = True),
                "workspace_file_content": attr.string(),
            },
        ),
    },
)
