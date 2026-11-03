"""Bootstrap the Zig worker without depending on the worker it is building."""

def _compile(ctx, is_test):
    toolchain = ctx.toolchains["@rules_zig//zig:toolchain_type"].zigtoolchaininfo
    executable = toolchain.zig_exe.file or toolchain.zig_exe.path
    output = ctx.actions.declare_file(ctx.label.name)
    args = ctx.actions.args()
    args.add("test" if is_test else "build-exe")
    if is_test:
        args.add("--test-no-exec")
    args.add(ctx.file.main)
    args.add_all(["-O", "Debug", "-fno-llvm", "-fno-lld"])
    args.add("--zig-lib-dir", toolchain.zig_lib.file or toolchain.zig_lib.path)
    args.add_all(["--cache-dir", toolchain.zig_cache, "--global-cache-dir", toolchain.zig_cache])
    args.add(output, format = "-femit-bin=%s")
    tool_files = [toolchain.validation]
    if toolchain.zig_lib.file:
        tool_files.append(toolchain.zig_lib.file)
    ctx.actions.run(
        executable = executable,
        arguments = [args],
        inputs = depset(ctx.files.srcs + [ctx.file.main] + tool_files),
        tools = [executable] if type(executable) == "File" else [],
        outputs = [output],
        mnemonic = "ZigWorkerBootstrap",
        toolchain = "@rules_zig//zig:toolchain_type",
    )
    runfiles = ctx.runfiles(files = ctx.files.data + tool_files + ([executable] if type(executable) == "File" else []))
    for dep in ctx.attr.data:
        runfiles = runfiles.merge(dep[DefaultInfo].default_runfiles)
    providers = [DefaultInfo(executable = output, runfiles = runfiles)]
    if is_test:
        providers.append(RunEnvironmentInfo(environment = {
            "ZIG_WORKER_TEST_COMPILER": executable.short_path if type(executable) == "File" else executable,
            "ZIG_WORKER_TEST_BINARY": ctx.executable.worker.short_path,
        }))
        providers[0] = DefaultInfo(executable = output, runfiles = runfiles.merge(ctx.attr.worker[DefaultInfo].default_runfiles).merge(ctx.runfiles(files = [ctx.executable.worker])))
    return providers

def _binary_impl(ctx):
    return _compile(ctx, False)

def _test_impl(ctx):
    return _compile(ctx, True)

_ATTRS = {
    "main": attr.label(allow_single_file = [".zig"], mandatory = True),
    "srcs": attr.label_list(allow_files = [".zig"]),
    "data": attr.label_list(allow_files = True),
}

worker_binary = rule(
    implementation = _binary_impl,
    attrs = _ATTRS,
    executable = True,
    toolchains = ["@rules_zig//zig:toolchain_type"],
)

worker_test = rule(
    implementation = _test_impl,
    attrs = _ATTRS | {"worker": attr.label(executable = True, cfg = "exec", mandatory = True)},
    test = True,
    toolchains = ["@rules_zig//zig:toolchain_type"],
)
