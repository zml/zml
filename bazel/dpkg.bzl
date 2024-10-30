def _packages_to_dict(txt):
    packages = {}
    current_pkg = {}
    for line in txt.splitlines():
        if line == "":
            if current_pkg:
                pkg_name = current_pkg["Package"]
                pkg_version = current_pkg["Version"]
                if pkg_name not in packages:
                    packages[pkg_name] = {}
                packages[pkg_name][pkg_version] = struct(**current_pkg)
            current_pkg = {}
            continue
        if line.startswith(" "):
            current_pkg[key] += line
            continue
        split = line.split(": ", 1)
        key = split[0]
        value = len(split) > 1 and split[1] or ""
        current_pkg[key] = value
    return packages

def _read_packages(mctx, label):
    data = mctx.read(Label(label))
    return _packages_to_dict(data)

dpkg = struct(
    read_packages = _read_packages,
)
