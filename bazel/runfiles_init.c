extern void _bazel_runfiles_init_impl(int argc, char **argv, char **envp);

// Preserve the existing startup ABI. Registration happens in C;
// runfiles discovery and initialization remain in Zig.
// glibc and Darwin's dyld pass argc/argv/envp to constructors; musl does not.
// The original Zig initializer already made this assumption to read argv[0].
//
// Let C emit the platform's constructor registration. Zig's incremental ELF
// linker does not yet handle .init_array's section type, flags, and size correctly.
// TODO: Revert this workaround once Zig issue #37453 is fixed:
// https://codeberg.org/ziglang/zig/issues/37453
__attribute__((constructor))
void _bazel_runfiles_init(int argc, char **argv, char **envp) {
    _bazel_runfiles_init_impl(argc, argv, envp);
}
