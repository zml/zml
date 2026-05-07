load("@llvm-21-kvx-raw//utils/bazel:configure.bzl", "llvm_configure")
load("//third_party/zig:repo.bzl", zig_source = "repo")

def _zig_impl(mctx):
    llvm_configure(
        name = "llvm-project",
        targets = [
            "AArch64",
            "AMDGPU",
            "ARM",
            "AVR",
            "BPF",
            "Hexagon",
            "KVX",
            "Lanai",
            "LoongArch",
            "Mips",
            "MSP430",
            "NVPTX",
            "PowerPC",
            "RISCV",
            "Sparc",
            "SPIRV",
            "SystemZ",
            "VE",
            "WebAssembly",
            "X86",
            "XCore",
        ],
    )
    zig_source()

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["zig"],
        root_module_direct_dev_deps = [],
    )

zig = module_extension(
    implementation = _zig_impl,
)
